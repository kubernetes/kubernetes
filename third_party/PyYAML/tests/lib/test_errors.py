
import yaml, test_emitter

def test_loader_error(error_filename, verbose=False):
    try:
        list(yaml.load_all(open(error_filename, 'rb')))
    except yaml.YAMLError, exc:
        if verbose:
            print "%s:" % exc.__class__.__name__, exc
    else:
        raise AssertionError("expected an exception")

test_loader_error.unittest = ['.loader-error']

def test_loader_error_string(error_filename, verbose=False):
    try:
        list(yaml.load_all(open(error_filename, 'rb').read()))
    except yaml.YAMLError, exc:
        if verbose:
            print "%s:" % exc.__class__.__name__, exc
    else:
        raise AssertionError("expected an exception")

test_loader_error_string.unittest = ['.loader-error']

def test_loader_error_single(error_filename, verbose=False):
    try:
        yaml.load(open(error_filename, 'rb').read())
    except yaml.YAMLError, exc:
        if verbose:
            print "%s:" % exc.__class__.__name__, exc
    else:
        raise AssertionError("expected an exception")

test_loader_error_single.unittest = ['.single-loader-error']

def test_emitter_error(error_filename, verbose=False):
    events = list(yaml.load(open(error_filename, 'rb'),
                    Loader=test_emitter.EventsLoader))
    try:
        yaml.emit(events)
    except yaml.YAMLError, exc:
        if verbose:
            print "%s:" % exc.__class__.__name__, exc
    else:
        raise AssertionError("expected an exception")

test_emitter_error.unittest = ['.emitter-error']

def test_dumper_error(error_filename, verbose=False):
    code = open(error_filename, 'rb').read()
    try:
        import yaml
        from StringIO import StringIO
        exec code
    except yaml.YAMLError, exc:
        if verbose:
            print "%s:" % exc.__class__.__name__, exc
    else:
        raise AssertionError("expected an exception")

test_dumper_error.unittest = ['.dumper-error']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

