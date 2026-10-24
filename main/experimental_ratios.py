"""Channel-by-channel proton-lifetime comparisons with experimental limits."""

from __future__ import annotations

from collections.abc import Mapping

from bounds import get_experimental_limits


BOUND_CHANNEL_ALIASES = {
    "p->pi+ nubar": "p->pi+ nu",
    "p->K+ nubar": "p->K+ nu",
}


def compare_lifetimes_to_bounds(lifetimes_years: Mapping[str, float]) -> tuple[dict, dict]:
    """Return exact per-channel ratios and an overall experimental summary."""
    comparisons = {}
    experimental_limits = get_experimental_limits()
    for channel, lifetime in lifetimes_years.items():
        if channel == "total":
            continue
        bound_channel = BOUND_CHANNEL_ALIASES.get(channel, channel)
        limits = experimental_limits.get(bound_channel)
        predicted = float(lifetime)
        if limits is None:
            comparisons[channel] = {
                "bound_channel": bound_channel,
                "predicted_lifetime_years": predicted,
                "current_bound_years": None,
                "theory_over_current_bound": None,
                "passes_current_bound": None,
                "future_sensitivity_years": None,
                "theory_over_future_sensitivity": None,
                "within_future_sensitivity": None,
            }
            continue

        current = float(limits["current"])
        future = limits.get("future")
        comparisons[channel] = {
            "bound_channel": bound_channel,
            "predicted_lifetime_years": predicted,
            "current_bound_years": current,
            "theory_over_current_bound": predicted / current,
            "passes_current_bound": predicted >= current,
            "future_sensitivity_years": None if future is None else float(future),
            "theory_over_future_sensitivity": None if future is None else predicted / float(future),
            "within_future_sensitivity": None if future is None else predicted < float(future),
        }

    bounded = {
        channel: data for channel, data in comparisons.items()
        if data["theory_over_current_bound"] is not None
    }
    excluded_channels = [
        channel for channel, data in bounded.items() if not data["passes_current_bound"]
    ]
    testable_channels = [
        channel for channel, data in bounded.items()
        if data["passes_current_bound"] and data["within_future_sensitivity"] is True
    ]
    most_constraining = min(
        bounded,
        key=lambda channel: bounded[channel]["theory_over_current_bound"],
        default=None,
    )

    if excluded_channels:
        classification = "excluded"
        reason = "At least one predicted partial lifetime is below its current experimental lower bound."
    elif testable_channels:
        classification = "testable"
        reason = "All predicted partial lifetimes pass current bounds, and at least one lies within a stated future sensitivity."
    else:
        classification = "safe"
        reason = "All predicted partial lifetimes pass current bounds, and none lies within a stated future sensitivity."

    summary = {
        "ratio_definition": "predicted partial lifetime / current experimental lower bound",
        "ratio_interpretation": "ratio < 1: excluded by that channel; ratio >= 1: passes that current bound",
        "classification": classification,
        "classification_reason": reason,
        "excluded_channels": excluded_channels,
        "testable_channels": testable_channels,
        "most_constraining_channel": most_constraining,
        "most_constraining_ratio": (
            None if most_constraining is None
            else bounded[most_constraining]["theory_over_current_bound"]
        ),
    }
    return comparisons, summary
