def convert_bools_in_dict(d):
    def str_to_bool(s):
        if isinstance(s, str):
            if s.lower() in ['true', '1', 'yes', 'y']:
                return True
            elif s.lower() in ['false', '0', 'no', 'n']:
                return False
        return s  # Return the original value if it's not a recognized boolean string

    return {k: str_to_bool(v) for k, v in d.items()}