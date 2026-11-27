"""Explicit experimental repair of detached Critic gradients and Actor saturation.

The archived implementation remains untouched. This variant has a separate
checkpoint identity and must start in a fresh run; it cannot revive a saturated
actor by silently loading an old run. Observation, raw replay actions, rewards
and episode collection are preserved. Learning stabilization is explicit and
versioned separately from the archived implementation.
"""
import torch
import torch.nn.functional as F

from original_td3 import OriginalTD3
from original_td3_models import Critic as ArchivedCritic
from td3_fusion import device


IMPLEMENTATION_ID = 'cimurs_common_scene_td3_repair_v2'


class GradientFixedCritic(ArchivedCritic):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # The archived forward never adds these two state-projection biases.
        # Keep exactly the same function, making unused parameters explicit.
        self.layer_2_s.bias.requires_grad_(False)
        self.layer_5_s.bias.requires_grad_(False)

    def forward(self, state, action):
        s1 = F.relu(self.layer_1(state))
        s1 = F.relu(F.linear(s1, self.layer_2_s.weight) + self.layer_2_a(action))
        q1 = self.layer_3(s1)
        s2 = F.relu(self.layer_4(state))
        s2 = F.relu(F.linear(s2, self.layer_5_s.weight) + self.layer_5_a(action))
        return q1, self.layer_6(s2)


class GradientFixedTD3(OriginalTD3):
    def __init__(self, vec_dim, action_dim=2, log_dir=None, actor_lr=1e-4,
                 critic_lr=3e-4, actor_warmup_updates=500,
                 logit_limit=2., logit_penalty=.01, **kwargs):
        super().__init__(vec_dim, action_dim, log_dir=log_dir, **kwargs)
        # Do not alter the initial models or consume extra exploration RNG.
        with torch.random.fork_rng(devices=[]):
            critic = GradientFixedCritic(vec_dim, action_dim).to(device)
            target = GradientFixedCritic(vec_dim, action_dim).to(device)
        critic.load_state_dict(self.critic.state_dict())
        target.load_state_dict(self.critic_target.state_dict())
        self.critic, self.critic_target = critic, target
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            (p for p in critic.parameters() if p.requires_grad),lr=critic_lr)
        self.actor_warmup_updates=int(actor_warmup_updates)
        self.logit_limit,self.logit_penalty=float(logit_limit),float(logit_penalty)
        self.critic_update_count=self.actor_update_count=0
        self.learning_protocol=dict(version='td3_repair_v2',actor_lr=float(actor_lr),
            critic_lr=float(critic_lr),actor_warmup_updates=self.actor_warmup_updates,
            logit_limit=self.logit_limit,logit_penalty=self.logit_penalty,
            delay_counter='cumulative_critic_updates')

    def train(self,replay_buffer,iterations=1,batch_size=40,discount=.99999,
              tau=.005,policy_noise=.2,noise_clip=.5,policy_freq=2):
        if iterations<=0 or replay_buffer.size()==0:return
        if policy_freq<1:raise ValueError('policy_freq must be positive')
        critic_losses=[];actor_losses=[];penalties=[]
        trainable=[p for p in self.critic.parameters() if p.requires_grad]
        for _ in range(int(iterations)):
            state,_,action,reward,done,next_state,_=replay_buffer.sample_batch(batch_size)
            state=torch.as_tensor(state,dtype=torch.float32,device=device)
            next_state=torch.as_tensor(next_state,dtype=torch.float32,device=device)
            action=torch.as_tensor(action,dtype=torch.float32,device=device)
            reward=torch.as_tensor(reward,dtype=torch.float32,device=device)
            done=torch.as_tensor(done,dtype=torch.float32,device=device)
            with torch.no_grad():
                noise=(torch.randn_like(action)*policy_noise).clamp(-noise_clip,noise_clip)
                next_action=(self.actor_target(next_state)+noise).clamp(-1,1)
                tq1,tq2=self.critic_target(next_state,next_action)
                target=reward+(1-done)*discount*torch.minimum(tq1,tq2)
            q1,q2=self.critic(state,action)
            critic_loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,10.,error_if_nonfinite=True)
            self.critic_optimizer.step()
            self.critic_update_count+=1
            critic_losses.append(float(critic_loss.detach()))
            if self.critic_update_count%policy_freq==0:
                if self.critic_update_count>self.actor_warmup_updates:
                    # Critic supplies dQ/da without accumulating a second set of
                    # Critic parameter gradients during the policy update.
                    for p in trainable:p.requires_grad_(False)
                    try:
                        hidden=F.relu(self.actor.layer_1(state))
                        logits=self.actor.layer_3(F.relu(self.actor.layer_2(hidden)))
                        penalty=F.relu(logits.abs()-self.logit_limit).square().mean()
                        actor_loss=-self.critic(state,torch.tanh(logits))[0].mean()+self.logit_penalty*penalty
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
                        self.actor_optimizer.step()
                        self.actor_update_count+=1
                        actor_losses.append(float(actor_loss.detach()))
                        penalties.append(float(penalty.detach()))
                    finally:
                        for p in trainable:p.requires_grad_(True)
                with torch.no_grad():
                    for source,target_parameter in zip(self.actor.parameters(),self.actor_target.parameters()):
                        target_parameter.mul_(1-tau).add_(source,alpha=tau)
                    for source,target_parameter in zip(self.critic.parameters(),self.critic_target.parameters()):
                        target_parameter.mul_(1-tau).add_(source,alpha=tau)
        self.iter_count+=1
        self.writer.add_scalar('repair/critic_loss',sum(critic_losses)/len(critic_losses),self.critic_update_count)
        if actor_losses:
            self.writer.add_scalar('repair/actor_loss',sum(actor_losses)/len(actor_losses),self.critic_update_count)
            self.writer.add_scalar('repair/logit_penalty',sum(penalties)/len(penalties),self.critic_update_count)

    def save_training_state(self, path, metadata=None):
        torch.save(dict(
            implementation_id=IMPLEMENTATION_ID,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            actor_target=self.actor_target.state_dict(), critic_target=self.critic_target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            iter_count=self.iter_count,critic_update_count=self.critic_update_count,
            actor_update_count=self.actor_update_count,learning_protocol=self.learning_protocol,
            metadata=metadata or {},
        ), path)

    def load_training_state(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get('implementation_id') != IMPLEMENTATION_ID:
            raise ValueError('td3_gradfix requires its own fresh checkpoints; archived runs are incompatible')
        if checkpoint.get('learning_protocol')!=self.learning_protocol:
            raise ValueError('td3_gradfix checkpoint learning protocol differs from this run')
        for name in ('actor', 'critic', 'actor_target', 'critic_target'):
            getattr(self, name).load_state_dict(checkpoint[name])
        if load_optimizers:
            for name in ('actor_optimizer', 'critic_optimizer'):
                getattr(self, name).load_state_dict(checkpoint[name])
        self.iter_count = int(checkpoint.get('iter_count', 0))
        self.critic_update_count=int(checkpoint['critic_update_count'])
        self.actor_update_count=int(checkpoint['actor_update_count'])
        return checkpoint.get('metadata', {})
