config_data = {
    "train-mode": 1,          # Training mode enabled (1)
    "tower-seed": -1,         # Use a random tower on every reset
    "starting-floor": 0,      # Start at floor 0 (easier for early training)
    "total-floors": 10,       # Limit tower height to 10 floors (shorter episodes)
    "dense-reward": 1,        # Use dense rewards to provide more frequent feedback
    "lighting-type": 0,       # No realtime lighting for simpler visuals
    "visual-theme": 0,        # Use only the default theme for consistency
    "agent-perspective": 1,   # Third-person view (might offer a wider field of view)
    "allowed-rooms": 1,       # Only normal rooms (exclude key and puzzle rooms)
    "allowed-modules": 1,     # Only easy modules (avoid the full range of challenges)
    "allowed-floors": 0,      # Only straightforward floor layouts (no branching or circling)
    "default-theme": 0        # Set the default theme to Ancient (0)
}