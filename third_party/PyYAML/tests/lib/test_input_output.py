
import yaml
import codecs, StringIO, tempfile, os, os.path

def _unicode_open(file, encoding, errors='strict'):
    info = codecs.lookup(encoding)
    if isinstance(info, tuple):
        reader = info[2]
        writer = info[3]
    else:
        reader = info.streamreader
        writer = info.streamwriter
    srw = codecs.StreamReaderWriter(file, reader, writer, errors)
    srw.encoding = encoding
    return srw

def test_unicode_input(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    value = ' '.join(data.split())
    output = yaml.load(_unicode_open(StringIO.StringIO(data.encode('utf-8')), 'utf-8'))
    assert output == value, (output, value)
    for input in [data, data.encode('utf-8'),
                    codecs.BOM_UTF8+data.encode('utf-8'),
                    codecs.BOM_UTF16_BE+data.encode('utf-16-be'),
                    codecs.BOM_UTF16_LE+data.encode('utf-16-le')]:
        if verbose:
            print "INPUT:", repr(input[:10]), "..."
        output = yaml.load(input)
        assert output == value, (output, value)
        output = yaml.load(StringIO.StringIO(input))
        assert output == value, (output, value)

test_unicode_input.unittest = ['.unicode']

def test_unicode_input_errors(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    for input in [data.encode('latin1', 'ignore'),
                    data.encode('utf-16-be'), data.encode('utf-16-le'),
                    codecs.BOM_UTF8+data.encode('utf-16-be'),
                    codecs.BOM_UTF16_BE+data.encode('utf-16-le'),
                    codecs.BOM_UTF16_LE+data.encode('utf-8')+'!']:
        try:
            yaml.load(input)
        except yaml.YAMLError, exc:
            if verbose:
                print exc
        else:
            raise AssertionError("expected an exception")
        try:
            yaml.load(StringIO.StringIO(input))
        except yaml.YAMLError, exc:
            if verbose:
                print exc
        else:
            raise AssertionError("expected an exception")

test_unicode_input_errors.unittest = ['.unicode']

def test_unicode_output(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    value = ' '.join(data.split())
    for allow_unicode in [False, True]:
        data1 = yaml.dump(value, allow_unicode=allow_unicode)
        for encoding in [None, 'utf-8', 'utf-16-be', 'utf-16-le']:
            stream = StringIO.StringIO()
            yaml.dump(value, _unicode_open(stream, 'utf-8'), encoding=encoding, allow_unicode=allow_unicode)
            data2 = stream.getvalue()
            data3 = yaml.dump(value, encoding=encoding, allow_unicode=allow_unicode)
            stream = StringIO.StringIO()
            yaml.dump(value, stream, encoding=encoding, allow_unicode=allow_unicode)
            data4 = stream.getvalue()
            for copy in [data1, data2, data3, data4]:
                if allow_unicode:
                    try:
                        copy[4:].encode('ascii')
                    except (UnicodeDecodeError, UnicodeEncodeError), exc:
                        if verbose:
                            print exc
                    else:
                        raise AssertionError("expected an exception")
                else:
                    copy[4:].encode('ascii')
            assert isinstance(data1, str), (type(data1), encoding)
            data1.decode('utf-8')
            assert isinstance(data2, str), (type(data2), encoding)
            data2.decode('utf-8')
            if encoding is None:
                assert isinstance(data3, unicode), (type(data3), encoding)
                assert isinstance(data4, unicode), (type(data4), encoding)
            else:
                assert isinstance(data3, str), (type(data3), encoding)
                data3.decode(encoding)
                assert isinstance(data4, str), (type(data4), encoding)
                data4.decode(encoding)

test_unicode_output.unittest = ['.unicode']

def test_file_output(unicode_filename, verbose=False):
    data = open(unicode_filename, 'rb').read().decode('utf-8')
    handle, filename = tempfile.mkstemp()
    os.close(handle)
    try:
        stream = StringIO.StringIO()
        yaml.dump(data, stream, allow_unicode=True)
        data1 = stream.getvalue()
        stream = open(filename, 'wb')
        yaml.dump(data, stream, allow_unicode=True)
        stream.close()
        data2 = open(filename, 'rb').read()
        stream = open(filename, 'wb')
        yaml.dump(data, stream, encoding='utf-16-le', allow_unicode=True)
        stream.close()
        data3 = open(filename, 'rb').read().decode('utf-16-le')[1:].encode('utf-8')
        stream = _unicode_open(open(filename, 'wb'), 'utf-8')
        yaml.dump(data, stream, allow_unicode=True)
        stream.close()
        data4 = open(filename, 'rb').read()
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
            input = (u'\ufeff'+input).encode(encoding)
        output1 = yaml.emit(yaml.parse(input), allow_unicode=True)
        stream = StringIO.StringIO()
        yaml.emit(yaml.parse(input), _unicode_open(stream, 'utf-8'),
                            allow_unicode=True)
        output2 = stream.getvalue()
        if encoding is None:
            assert isinstance(output1, unicode), (type(output1), encoding)
        else:
            assert isinstance(output1, str), (type(output1), encoding)
            output1.decode(encoding)
        assert isinstance(output2, str), (type(output2), encoding)
        output2.decode('utf-8')

test_unicode_transfer.unittest = ['.unicode']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

