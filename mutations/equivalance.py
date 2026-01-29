def is_equivalent(base_row, mutant_row, eps=1e-3):
    for key in ["DSI", "CPR", "FSI", "FVR"]:
        if abs(base_row[key] - mutant_row[key]) > eps:
            return False
    return True
