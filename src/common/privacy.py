"""Privacy constants shared by every service (defense in depth: each layer enforces k >= 5)."""
MIN_GROUP_SIZE = 5
PRIVACY_REFUSAL = "Privacy Guardrail Triggered: Minimum aggregate group size is 5"
