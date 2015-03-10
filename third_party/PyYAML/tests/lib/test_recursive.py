
import yaml

class AnInstance:

    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    def __repr__(self):
        try:
            return "%s(foo=%r, bar=%r)" % (self.__class__.__name__,
                    self.foo, self.bar)
        except RuntimeError:
            return "%s(foo=..., bar=...)" % self.__class__.__name__

class AnInstanceWithState(AnInstance):

    def __getstate__(self):
        return {'attributes': [self.foo, self.bar]}

    def __setstate__(self, state):
        self.foo, self.bar = state['attributes']

def test_recursive(recursive_filename, verbose=False):
    exec open(recursive_filename, 'rb').read()
    value1 = value
    output1 = None
    value2 = None
    output2 = None
    try:
        output1 = yaml.dump(value1)
        value2 = yaml.load(output1)
        output2 = yaml.dump(value2)
        assert output1 == output2, (output1, output2)
    finally:
        if verbose:
            #print "VALUE1:", value1
            #print "VALUE2:", value2
            print "OUTPUT1:"
            print output1
            print "OUTPUT2:"
            print output2

test_recursive.unittest = ['.recursive']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

