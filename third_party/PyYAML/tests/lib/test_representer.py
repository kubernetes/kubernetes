
import yaml
import test_constructor
import pprint

def test_representer_types(code_filename, verbose=False):
    test_constructor._make_objects()
    for allow_unicode in [False, True]:
        for encoding in ['utf-8', 'utf-16-be', 'utf-16-le']:
            native1 = test_constructor._load_code(open(code_filename, 'rb').read())
            native2 = None
            try:
                output = yaml.dump(native1, Dumper=test_constructor.MyDumper,
                            allow_unicode=allow_unicode, encoding=encoding)
                native2 = yaml.load(output, Loader=test_constructor.MyLoader)
                try:
                    if native1 == native2:
                        continue
                except TypeError:
                    pass
                value1 = test_constructor._serialize_value(native1)
                value2 = test_constructor._serialize_value(native2)
                if verbose:
                    print "SERIALIZED NATIVE1:"
                    print value1
                    print "SERIALIZED NATIVE2:"
                    print value2
                assert value1 == value2, (native1, native2)
            finally:
                if verbose:
                    print "NATIVE1:"
                    pprint.pprint(native1)
                    print "NATIVE2:"
                    pprint.pprint(native2)
                    print "OUTPUT:"
                    print output

test_representer_types.unittest = ['.code']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

