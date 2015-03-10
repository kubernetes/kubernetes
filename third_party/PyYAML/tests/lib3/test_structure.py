
import yaml, canonical
import pprint

def _convert_structure(loader):
    if loader.check_event(yaml.ScalarEvent):
        event = loader.get_event()
        if event.tag or event.anchor or event.value:
            return True
        else:
            return None
    elif loader.check_event(yaml.SequenceStartEvent):
        loader.get_event()
        sequence = []
        while not loader.check_event(yaml.SequenceEndEvent):
            sequence.append(_convert_structure(loader))
        loader.get_event()
        return sequence
    elif loader.check_event(yaml.MappingStartEvent):
        loader.get_event()
        mapping = []
        while not loader.check_event(yaml.MappingEndEvent):
            key = _convert_structure(loader)
            value = _convert_structure(loader)
            mapping.append((key, value))
        loader.get_event()
        return mapping
    elif loader.check_event(yaml.AliasEvent):
        loader.get_event()
        return '*'
    else:
        loader.get_event()
        return '?'

def test_structure(data_filename, structure_filename, verbose=False):
    nodes1 = []
    nodes2 = eval(open(structure_filename, 'r').read())
    try:
        loader = yaml.Loader(open(data_filename, 'rb'))
        while loader.check_event():
            if loader.check_event(yaml.StreamStartEvent, yaml.StreamEndEvent,
                                yaml.DocumentStartEvent, yaml.DocumentEndEvent):
                loader.get_event()
                continue
            nodes1.append(_convert_structure(loader))
        if len(nodes1) == 1:
            nodes1 = nodes1[0]
        assert nodes1 == nodes2, (nodes1, nodes2)
    finally:
        if verbose:
            print("NODES1:")
            pprint.pprint(nodes1)
            print("NODES2:")
            pprint.pprint(nodes2)

test_structure.unittest = ['.data', '.structure']

def _compare_events(events1, events2, full=False):
    assert len(events1) == len(events2), (len(events1), len(events2))
    for event1, event2 in zip(events1, events2):
        assert event1.__class__ == event2.__class__, (event1, event2)
        if isinstance(event1, yaml.AliasEvent) and full:
            assert event1.anchor == event2.anchor, (event1, event2)
        if isinstance(event1, (yaml.ScalarEvent, yaml.CollectionStartEvent)):
            if (event1.tag not in [None, '!'] and event2.tag not in [None, '!']) or full:
                assert event1.tag == event2.tag, (event1, event2)
        if isinstance(event1, yaml.ScalarEvent):
            assert event1.value == event2.value, (event1, event2)

def test_parser(data_filename, canonical_filename, verbose=False):
    events1 = None
    events2 = None
    try:
        events1 = list(yaml.parse(open(data_filename, 'rb')))
        events2 = list(yaml.canonical_parse(open(canonical_filename, 'rb')))
        _compare_events(events1, events2)
    finally:
        if verbose:
            print("EVENTS1:")
            pprint.pprint(events1)
            print("EVENTS2:")
            pprint.pprint(events2)

test_parser.unittest = ['.data', '.canonical']

def test_parser_on_canonical(canonical_filename, verbose=False):
    events1 = None
    events2 = None
    try:
        events1 = list(yaml.parse(open(canonical_filename, 'rb')))
        events2 = list(yaml.canonical_parse(open(canonical_filename, 'rb')))
        _compare_events(events1, events2, full=True)
    finally:
        if verbose:
            print("EVENTS1:")
            pprint.pprint(events1)
            print("EVENTS2:")
            pprint.pprint(events2)

test_parser_on_canonical.unittest = ['.canonical']

def _compare_nodes(node1, node2):
    assert node1.__class__ == node2.__class__, (node1, node2)
    assert node1.tag == node2.tag, (node1, node2)
    if isinstance(node1, yaml.ScalarNode):
        assert node1.value == node2.value, (node1, node2)
    else:
        assert len(node1.value) == len(node2.value), (node1, node2)
        for item1, item2 in zip(node1.value, node2.value):
            if not isinstance(item1, tuple):
                item1 = (item1,)
                item2 = (item2,)
            for subnode1, subnode2 in zip(item1, item2):
                _compare_nodes(subnode1, subnode2)

def test_composer(data_filename, canonical_filename, verbose=False):
    nodes1 = None
    nodes2 = None
    try:
        nodes1 = list(yaml.compose_all(open(data_filename, 'rb')))
        nodes2 = list(yaml.canonical_compose_all(open(canonical_filename, 'rb')))
        assert len(nodes1) == len(nodes2), (len(nodes1), len(nodes2))
        for node1, node2 in zip(nodes1, nodes2):
            _compare_nodes(node1, node2)
    finally:
        if verbose:
            print("NODES1:")
            pprint.pprint(nodes1)
            print("NODES2:")
            pprint.pprint(nodes2)

test_composer.unittest = ['.data', '.canonical']

def _make_loader():
    global MyLoader

    class MyLoader(yaml.Loader):
        def construct_sequence(self, node):
            return tuple(yaml.Loader.construct_sequence(self, node))
        def construct_mapping(self, node):
            pairs = self.construct_pairs(node)
            pairs.sort(key=(lambda i: str(i)))
            return pairs
        def construct_undefined(self, node):
            return self.construct_scalar(node)

    MyLoader.add_constructor('tag:yaml.org,2002:map', MyLoader.construct_mapping)
    MyLoader.add_constructor(None, MyLoader.construct_undefined)

def _make_canonical_loader():
    global MyCanonicalLoader

    class MyCanonicalLoader(yaml.CanonicalLoader):
        def construct_sequence(self, node):
            return tuple(yaml.CanonicalLoader.construct_sequence(self, node))
        def construct_mapping(self, node):
            pairs = self.construct_pairs(node)
            pairs.sort(key=(lambda i: str(i)))
            return pairs
        def construct_undefined(self, node):
            return self.construct_scalar(node)

    MyCanonicalLoader.add_constructor('tag:yaml.org,2002:map', MyCanonicalLoader.construct_mapping)
    MyCanonicalLoader.add_constructor(None, MyCanonicalLoader.construct_undefined)

def test_constructor(data_filename, canonical_filename, verbose=False):
    _make_loader()
    _make_canonical_loader()
    native1 = None
    native2 = None
    try:
        native1 = list(yaml.load_all(open(data_filename, 'rb'), Loader=MyLoader))
        native2 = list(yaml.load_all(open(canonical_filename, 'rb'), Loader=MyCanonicalLoader))
        assert native1 == native2, (native1, native2)
    finally:
        if verbose:
            print("NATIVE1:")
            pprint.pprint(native1)
            print("NATIVE2:")
            pprint.pprint(native2)

test_constructor.unittest = ['.data', '.canonical']

if __name__ == '__main__':
    import test_appliance
    test_appliance.run(globals())

