
import yaml
import pprint

def test_implicit_resolver(data_filename, detect_filename, verbose=False):
    correct_tag = None
    node = None
    try:
        correct_tag = open(detect_filename, 'r').read().strip()
        node = yaml.compose(open(data_filename, 'rb'))
        assert isinstance(node, yaml.SequenceNode), node
        for scalar in node.value:
            assert isinstance(scalar, yaml.ScalarNode), scalar
            assert scalar.tag == correct_tag, (scalar.tag, correct_tag)
    finally:
        if verbose:
            print("CORRECT TAG:", correct_tag)
            if hasattr(node, 'value'):
                print("CHILDREN:")
                pprint.pprint(node.value)

test_implicit_resolver.unittest = ['.data', '.detect']

def _make_path_loader_and_dumper():
    global MyLoader, MyDumper

    class MyLoader(yaml.Loader):
        pass
    class MyDumper(yaml.Dumper):
        pass

    yaml.add_path_resolver('!root', [],
            Loader=MyLoader, Dumper=MyDumper)
    yaml.add_path_resolver('!root/scalar', [], str,
            Loader=MyLoader, Dumper=MyDumper)
    yaml.add_path_resolver('!root/key11/key12/*', ['key11', 'key12'],
            Loader=MyLoader, Dumper=MyDumper)
    yaml.add_path_resolver('!root/key21/1/*', ['key21', 1],
            Loader=MyLoader, Dumper=MyDumper)
    yaml.add_path_resolver('!root/key31/*/*/key14/map', ['key31', None, None, 'key14'], dict,
            Loader=MyLoader, Dumper=MyDumper)

    return MyLoader, MyDumper

def _convert_node(node):
    if isinstance(node, yaml.ScalarNode):
        return (node.tag, node.value)
    elif isinstance(node, yaml.SequenceNode):
        value = []
        for item in node.value:
            value.append(_convert_node(item))
        return (node.tag, value)
    elif isinstance(node, yaml.MappingNode):
        value = []
        for key, item in node.value:
            value.append((_convert_node(key), _convert_node(item)))
        return (node.tag, value)

def test_path_resolver_loader(data_filename, path_filename, verbose=False):
    _make_path_loader_and_dumper()
    nodes1 = list(yaml.compose_all(open(data_filename, 'rb').read(), Loader=MyLoader))
    nodes2 = list(yaml.compose_all(open(path_filename, 'rb').read()))
    try:
        for node1, node2 in zip(nodes1, nodes2):
            data1 = _convert_node(node1)
            data2 = _convert_node(node2)
            assert data1 == data2, (data1, data2)
    finally:
        if verbose:
            print(yaml.serialize_all(nodes1))

test_path_resolver_loader.unittest = ['.data', '.path']

def test_path_resolver_dumper(data_filename, path_filename, verbose=False):
    _make_path_loader_and_dumper()
    for filename in [data_filename, path_filename]:
        output = yaml.serialize_all(yaml.compose_all(open(filename, 'rb')), Dumper=MyDumper)
        if verbose:
            print(output)
        nodes1 = yaml.compose_all(output)
        nodes2 = yaml.compose_all(open(data_filename, 'rb'))
        for node1, node2 in zip(nodes1, nodes2):
            data1 = _convert_node(node1)
            data2 = _convert_node(node2)
            assert data1 == data2, (data1, data2)

test_path_resolver_dumper.unittest = ['.data', '.path']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

