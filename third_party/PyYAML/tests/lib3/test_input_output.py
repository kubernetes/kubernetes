
import yaml
import codecs, io, tempfile, os, os.path

def test_unicode_input(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    value = ' '.join(data.split())
    output = yaml.load(data)
    assert output == value, (output, value)
    output = yaml.load(io.StringIO(data))
    assert output == value, (output, value)
    for input in [data.encode('utf-8'),
                    codecs.BOM_UTF8+data.encode('utf-8'),
                    codecs.BOM_UTF16_BE+data.encode('utf-16-be'),
                    codecs.BOM_UTF16_LE+data.encode('utf-16-le')]:
        if verbose:
            print("INPUT:", repr(input[:10]), "...")
        output = yaml.load(input)
        assert output == value, (output, value)
        output = yaml.load(io.BytesIO(input))
        assert output == value, (output, value)

test_unicode_input.unittest = ['.unicode']

def test_unicode_input_errors(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    for input in [data.encode('latin1', 'ignore'),
                    data.encode('utf-16-be'), data.encode('utf-16-le'),
                    codecs.BOM_UTF8+data.encode('utf-16-be'),
                    codecs.BOM_UTF16_BE+data.encode('utf-16-le'),
                    codecs.BOM_UTF16_LE+data.encode('utf-8')+b'!']:
        try:
            yaml.load(input)
        except yaml.YAMLError as exc:
            if verbose:
                print(exc)
        else:
            raise AssertionError("expected an exception")
        try:
            yaml.load(io.BytesIO(input))
        except yaml.YAMLError as exc:
            if verbose:
                print(exc)
        else:
            raise AssertionError("expected an exception")

test_unicode_input_errors.unittest = ['.unicode']

def test_unicode_output(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    value = ' '.join(data.split())
    for allow_unicode in [False, True]:
        data1 = yaml.dump(value, allow_unicode=allow_unicode)
        for encoding in [None, 'utf-8', 'utf-16-be', 'utf-16-le']:
            stream = io.StringIO()
            yaml.dump(value, stream, encoding=encoding, allow_unicode=allow_unicode)
            data2 = stream.getvalue()
            data3 = yaml.dump(value, encoding=encoding, allow_unicode=allow_unicode)
            if encoding is not None:
                assert isinstance(data3, bytes)
                data3 = data3.decode(encoding)
            stream = io.BytesIO()
            if encoding is None:
                try:
                    yaml.dump(value, stream, encoding=encoding, allow_unicode=allow_unicode)
                except TypeError as exc:
                    if verbose:
                        print(exc)
                    data4 = None
                else:
                    raise AssertionError("expected an exception")
            else:
                yaml.dump(value, stream, encoding=encoding, allow_unicode=allow_unicode)
                data4 = stream.getvalue()
                if verbose:
                    print("BYTES:", data4[:50])
                data4 = data4.decode(encoding)
            for copy in [data1, data2, data3, data4]:
                if copy is None:
                    continue
                assert isinstance(copy, str)
                if allow_unicode:
                    try:
                        copy[4:].encode('ascii')
                    except UnicodeEncodeError as exc:
                        if verbose:
                            print(exc)
                    else:
                        raise AssertionError("expected an exception")
                else:
                    copy[4:].encode('ascii')
            assert isinstance(data1, str), (type(data1), encoding)
            assert isinstance(data2, str), (type(data2), encoding)

test_unicode_output.unittest = ['.unicode']

def test_file_output(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    handle, filename = tempfile.mkstemp()
    os.close(handle)
    try:
        stream = io.StringIO()
        yaml.dump(data, stream, allow_unicode=True)
        data1 = stream.getvalue()
        stream = io.BytesIO()
        yaml.dump(data, stream, encoding='utf-16-le', allow_unicode=True)
        data2 = stream.getvalue().decode('utf-16-le')[1:]
        stream = open(filename, 'w', encoding='utf-16-le')
        yaml.dump(data, stream, allow_unicode=True)
        stream.close()
        data3 = open(filename, 'r', encoding='utf-16-le').read()
        stream = open(filename, 'wb')
        yaml.dump(data, stream, encoding='utf-8', allow_unicode=True)
        stream.close()
        data4 = open(filename, 'r', encoding='utf-8').read()
        assert data1 == data2, (data1, data2)
        assert data1 == data3, (data1, data3)
        assert data1 == data4, (data1, data4)
    finally:
        if os.path.exists(filename):
            os.unlink(filename)

test_file_output.unittest = ['.unicode']

def test_unicode_transfer(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    for encoding in [None, 'utf-8', 'utf-16-be', 'utf-16-le']:
        input = data
        if encoding is not None:
            input = ('\ufeff'+input).encode(encoding)
        output1 = yaml.emit(yaml.parse(input), allow_unicode=True)
        if encoding is None:
            stream = io.StringIO()
        else:
            stream = io.BytesIO()
        yaml.emit(yaml.parse(input), stream, allow_unicode=True)
        output2 = stream.getvalue()
        assert isinstance(output1, str), (type(output1), encoding)
        if encoding is None:
            assert isinstance(output2, str), (type(output1), encoding)
        else:
            assert isinstance(output2, bytes), (type(output1), encoding)
            output2.decode(encoding)

test_unicode_transfer.unittest = ['.unicode']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

