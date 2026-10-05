"""Lookup table: detection signal -> MITRE ATT&CK techniques. No logic lives here."""

TECHNIQUES = {
    "T1021": "Remote Services",
    "T1021.001": "Remote Services: Remote Desktop Protocol",
    "T1078": "Valid Accounts",
    "T1110": "Brute Force",
    "T1550.002": "Use Alternate Authentication Material: Pass the Hash",
}

# Rules (baseline_rules.RULES), Isolation Forest features (features.FEATURES)
# and Transformer fields (model_transformer.FIELDS) share one namespace.
SIGNAL_TO_TECHNIQUES = {
    # rules
    "fanout": ["T1021"],
    "first_pair": ["T1021", "T1078"],
    "off_hours": ["T1078"],
    "fail_then_success": ["T1110"],
    "ntlm_new_host": ["T1550.002", "T1021"],
    "rdp_new_host": ["T1021.001"],
    "new_src": ["T1078"],
    # window features
    "n_events": ["T1078"],
    "n_dst": ["T1021"],
    "n_src": ["T1078"],
    "n_new_pair": ["T1021", "T1078"],
    "n_new_src": ["T1078"],
    "n_fail": ["T1110"],
    "fail_ratio": ["T1110"],
    "n_ntlm": ["T1550.002"],
    "n_network": ["T1021"],
    "n_rdp": ["T1021.001"],
    "min_hour_freq": ["T1078"],
    # event fields
    "dst_comp": ["T1021"],
    "src_comp": ["T1078"],
    "auth_type": ["T1550.002"],
    "logon_type": ["T1021"],
    "auth_orientation": ["T1078"],
    "success": ["T1110"],
    "hod": ["T1078"],
}


def techniques_for(signals: list[str]) -> list[str]:
    """Sorted unique 'ID Name' strings for a list of signal names."""
    ids = sorted({t for s in signals for t in SIGNAL_TO_TECHNIQUES.get(s, [])})
    return [f"{t} {TECHNIQUES[t]}" for t in ids]


def markdown_table() -> str:
    rows = ["| Signal | ATT&CK |", "|---|---|"]
    rows += [f"| `{s}` | {', '.join(techniques_for([s]))} |" for s in SIGNAL_TO_TECHNIQUES]
    return "\n".join(rows)


if __name__ == "__main__":
    print(markdown_table())
