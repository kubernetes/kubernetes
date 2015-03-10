
import yaml.reader
import codecs

def _run_reader(data, verbose):
    try:
        stream = yaml.reader.Reader(data)
        while stream.peek() != u'\0':
            stream.forward()
    except yaml.reader.ReaderError, exc:
        if verbose:
            print exc
    else:
        raise AssertionError("expected an exception")

def test_stream_error(error_filename, verbose=False):
    _run_reader(open(error_filename, 'rb'), verbose)
    _run_reader(open(error_filename, 'rb').read(), verbose)
    for encoding in ['utf-8', 'utf-16-le', 'utf-16-be']:
        try:
            data = unicode(open(error_filename, 'rb').read(), encoding)
            break
        except UnicodeDecodeError:
            pass
    else:
        return
    _run_reader(data, verbose)
    _run_reader(codecs.open(error_filename, encoding=encoding), verbose)

test_stream_error.unittest = ['.stream-error']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

