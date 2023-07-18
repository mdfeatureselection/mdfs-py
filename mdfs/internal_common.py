def handle_error(result):
    if result._error is not None:
        raise Exception(result._error.decode('utf-8'))
    return result
