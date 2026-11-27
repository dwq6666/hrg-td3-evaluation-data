"""Safety authorization for the policy's independently learned pass signal."""


def authorized_pass_gate(requested_gate, front_class):
    """Allow override only when the currently sensed obstacle is passable-low."""
    return float(requested_gate) if front_class == "low" else 0.0


def unsafe_pass_request_cost(requested_gate, front_class):
    """Small training penalty for asking to pass a high barrier or pit."""
    return 0.05 * float(requested_gate) if front_class in ("high", "pit") else 0.0
