"""Choose the focus of a round using only measured local progress."""


def iteration_focus(iteration: int, has_valid_result: bool, num_agents: int) -> str:
    """Return baseline, mixed, improve, or explore; iteration is zero-based."""
    # Customization point: balance improving the best result against new approaches.
    if not has_valid_result:
        return "baseline"
    if num_agents > 1:
        return "mixed"
    return "improve" if iteration % 2 else "explore"
