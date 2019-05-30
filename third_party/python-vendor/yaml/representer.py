
__all__ = ['BaseRepresenter', 'SafeRepresenter', 'Representer',
    'RepresenterError']

from error import *
from nodes import *

import datetime

import sys, copy_reg, types

class RepresenterError(YAMLError):
    pass

class BaseRepresenter(object):

    yaml_representers = {}
    yaml_multi_representers = {}

    def __init__(self, default_style=None, default_flow_style=False, sort_keys=True):
        self.default_style = default_style
        self.default_flow_style = default_flow_style
        self.sort_keys = sort_keys
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def represent(self, data):
        node = self.represent_data(data)
        self.serialize(node)
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def get_classobj_bases(self, cls):
        bases = [cls]
        for base in cls.__bases__:
            bases.extend(self.get_classobj_bases(base))
        return bases

    def represent_data(self, data):
        if self.ignore_aliases(data):
            self.alias_key = None
        else:
            self.alias_key = id(data)
        if self.alias_key is not None:
            if self.alias_key in self.represented_objects:
                node = self.represented_objects[self.alias_key]
                #if node is None:
                #    raise RepresenterError("recursive objects are not allowed: %r" % data)
                return node
            #self.represented_objects[alias_key] = None
            self.object_keeper.append(data)
        data_types = type(data).__mro__
        if type(data) is types.InstanceType:
            data_types = self.get_classobj_bases(data.__class__)+list(data_types)
        if data_types[0] in self.yaml_representers:
            node = self.yaml_representers[data_types[0]](self, data)
        else:
            for data_type in data_types:
                if data_type in self.yaml_multi_representers:
                    node = self.yaml_multi_representers[data_type](self, data)
                    break
            else:
                if None in self.yaml_multi_representers:
                    node = self.yaml_multi_representers[None](self, data)
                elif None in self.yaml_representers:
                    node = self.yaml_representers[None](self, data)
                else:
                    node = ScalarNode(None, unicode(data))
        #if alias_key is not None:
        #    self.represented_objects[alias_key] = node
        return node

    def add_representer(cls, data_type, representer):
        if not 'yaml_representers' in cls.__dict__:
            cls.yaml_representers = cls.yaml_representers.copy()
        cls.yaml_representers[data_type] = representer
    add_representer = classmethod(add_representer)

    def add_multi_representer(cls, data_type, representer):
        if not 'yaml_multi_representers' in cls.__dict__:
            cls.yaml_multi_representers = cls.yaml_multi_representers.copy()
        cls.yaml_multi_representers[data_type] = representer
    add_multi_representer = classmethod(add_multi_representer)

    def represent_scalar(self, tag, value, style=None):
        if style is None:
            style = self.default_style
        node = ScalarNode(tag, value, style=style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        return node

    def represent_sequence(self, tag, sequence, flow_style=None):
        value = []
        node = SequenceNode(tag, value, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        for item in sequence:
            node_item = self.represent_data(item)
            if not (isinstance(node_item, ScalarNode) and not node_item.style):
                best_style = False
            value.append(node_item)
        if flow_style is None:
            if self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def represent_mapping(self, tag, mapping, flow_style=None):
        value = []
        node = MappingNode(tag, value, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        if hasattr(mapping, 'items'):
            mapping = mapping.items()
            if self.sort_keys:
                mapping.sort()
        for item_key, item_value in mapping:
            node_key = self.represent_data(item_key)
            node_value = self.represent_data(item_value)
            if not (isinstance(node_key, ScalarNode) and not node_key.style):
                best_style = False
            if not (isinstance(node_value, ScalarNode) and not node_value.style):
                best_style = False
            value.append((node_key, node_value))
        if flow_style is None:
            if self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def ignore_aliases(self, data):
        return False

class SafeRepresenter(BaseRepresenter):

    def ignore_aliases(self, data):
        if data is None:
            return True
        if isinstance(data, tuple) and data == ():
            return True
        if isinstance(data, (str, unicode, bool, int, float)):
            return True

    def represent_none(self, data):
        return self.represent_scalar(u'tag:yaml.org,2002:null',
                u'null')

    def represent_str(self, data):
        tag = None
        style = None
        try:
            data = unicode(data, 'ascii')
            tag = u'tag:yaml.org,2002:str'
        except UnicodeDecodeError:
            try:
                data = unicode(data, 'utf-8')
                tag = u'tag:yaml.org,2002:str'
            except UnicodeDecodeError:
                data = data.encode('base64')
                tag = u'tag:yaml.org,2002:binary'
                style = '|'
        return self.represent_scalar(tag, data, style=style)

    def represent_unicode(self, data):
        return self.represent_scalar(u'tag:yaml.org,2002:str', data)

    def represent_bool(self, data):
        if data:
            value = u'true'
        else:
            value = u'false'
        return self.represent_scalar(u'tag:yaml.org,2002:bool', value)

    def represent_int(self, data):
        return self.represent_scalar(u'tag:yaml.org,2002:int', unicode(data))

    def represent_long(self, data):
        return self.represent_scalar(u'tag:yaml.org,2002:int', unicode(data))

    inf_value = 1e300
    while repr(inf_value) != repr(inf_value*inf_value):
        inf_value *= inf_value

    def represent_float(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = u'.nan'
        elif data == self.inf_value:
            value = u'.inf'
        elif data == -self.inf_value:
            value = u'-.inf'
        else:
            value = unicode(repr(data)).lower()
            # Note that in some cases `repr(data)` represents a float number
            # without the decimal parts.  For instance:
            #   >>> repr(1e17)
            #   '1e17'
            # Unfortunately, this is not a valid float representation according
            # to the definition of the `!!float` tag.  We fix this by adding
            # '.0' before the 'e' symbol.
            if u'.' not in value and u'e' in value:
                value = value.replace(u'e', u'.0e', 1)
        return self.represent_scalar(u'tag:yaml.org,2002:float', value)

    def represent_list(self, data):
        #pairs = (len(data) > 0 and isinstance(data, list))
        #if pairs:
        #    for item in data:
        #        if not isinstance(item, tuple) or len(item) != 2:
        #            pairs = False
        #            break
        #if not pairs:
            return self.represent_sequence(u'tag:yaml.org,2002:seq', data)
        #value = []
        #for item_key, item_value in data:
        #    value.append(self.represent_mapping(u'tag:yaml.org,2002:map',
        #        [(item_key, item_value)]))
        #return SequenceNode(u'tag:yaml.org,2002:pairs', value)

    def represent_dict(self, data):
        return self.represent_mapping(u'tag:yaml.org,2002:map', data)

    def represent_set(self, data):
        value = {}
        for key in data:
            value[key] = None
        return self.represent_mapping(u'tag:yaml.org,2002:set', value)

    def represent_date(self, data):
        value = unicode(data.isoformat())
        return self.represent_scalar(u'tag:yaml.org,2002:timestamp', value)

    def represent_datetime(self, data):
        value = unicode(data.isoformat(' '))
        return self.represent_scalar(u'tag:yaml.org,2002:timestamp', value)

    def represent_yaml_object(self, tag, data, cls, flow_style=None):
        if hasattr(data, '__getstate__'):
            state = data.__getstate__()
        else:
            state = data.__dict__.copy()
        return self.represent_mapping(tag, state, flow_style=flow_style)

    def represent_undefined(self, data):
        raise RepresenterError("cannot represent an object", data)

SafeRepresenter.add_representer(type(None),
        SafeRepresenter.represent_none)

SafeRepresenter.add_representer(str,
        SafeRepresenter.represent_str)

SafeRepresenter.add_representer(unicode,
        SafeRepresenter.represent_unicode)

SafeRepresenter.add_representer(bool,
        SafeRepresenter.represent_bool)

SafeRepresenter.add_representer(int,
        SafeRepresenter.represent_int)

SafeRepresenter.add_representer(long,
        SafeRepresenter.represent_long)

SafeRepresenter.add_representer(float,
        SafeRepresenter.represent_float)

SafeRepresenter.add_representer(list,
        SafeRepresenter.represent_list)

SafeRepresenter.add_representer(tuple,
        SafeRepresenter.represent_list)

SafeRepresenter.add_representer(dict,
        SafeRepresenter.represent_dict)

SafeRepresenter.add_representer(set,
        SafeRepresenter.represent_set)

SafeRepresenter.add_representer(datetime.date,
        SafeRepresenter.represent_date)

SafeRepresenter.add_representer(datetime.datetime,
        SafeRepresenter.represent_datetime)

SafeRepresenter.add_representer(None,
        SafeRepresenter.represent_undefined)

class Representer(SafeRepresenter):

    def represent_str(self, data):
        tag = None
        style = None
        try:
            data = unicode(data, 'ascii')
            tag = u'tag:yaml.org,2002:str'
        except UnicodeDecodeError:
            try:
                data = unicode(data, 'utf-8')
                tag = u'tag:yaml.org,2002:python/str'
            except UnicodeDecodeError:
                data = data.encode('base64')
                tag = u'tag:yaml.org,2002:binary'
                style = '|'
        return self.represent_scalar(tag, data, style=style)

    def represent_unicode(self, data):
        tag = None
        try:
            data.encode('ascii')
            tag = u'tag:yaml.org,2002:python/unicode'
        except UnicodeEncodeError:
            tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data)

    def represent_long(self, data):
        tag = u'tag:yaml.org,2002:int'
        if int(data) is not data:
            tag = u'tag:yaml.org,2002:python/long'
        return self.represent_scalar(tag, unicode(data))

    def represent_complex(self, data):
        if data.imag == 0.0:
            data = u'%r' % data.real
        elif data.real == 0.0:
            data = u'%rj' % data.imag
        elif data.imag > 0:
            data = u'%r+%rj' % (data.real, data.imag)
        else:
            data = u'%r%rj' % (data.real, data.imag)
        return self.represent_scalar(u'tag:yaml.org,2002:python/complex', data)

    def represent_tuple(self, data):
        return self.represent_sequence(u'tag:yaml.org,2002:python/tuple', data)

    def represent_name(self, data):
        name = u'%s.%s' % (data.__module__, data.__name__)
        return self.represent_scalar(u'tag:yaml.org,2002:python/name:'+name, u'')

    def represent_module(self, data):
        return self.represent_scalar(
                u'tag:yaml.org,2002:python/module:'+data.__name__, u'')

    def represent_instance(self, data):
        # For instances of classic classes, we use __getinitargs__ and
        # __getstate__ to serialize the data.

        # If data.__getinitargs__ exists, the object must be reconstructed by
        # calling cls(**args), where args is a tuple returned by
        # __getinitargs__. Otherwise, the cls.__init__ method should never be
        # called and the class instance is created by instantiating a trivial
        # class and assigning to the instance's __class__ variable.

        # If data.__getstate__ exists, it returns the state of the object.
        # Otherwise, the state of the object is data.__dict__.

        # We produce either a !!python/object or !!python/object/new node.
        # If data.__getinitargs__ does not exist and state is a dictionary, we
        # produce a !!python/object node . Otherwise we produce a
        # !!python/object/new node.

        cls = data.__class__
        class_name = u'%s.%s' % (cls.__module__, cls.__name__)
        args = None
        state = None
        if hasattr(data, '__getinitargs__'):
            args = list(data.__getinitargs__())
        if hasattr(data, '__getstate__'):
            state = data.__getstate__()
        else:
            state = data.__dict__
        if args is None and isinstance(state, dict):
            return self.represent_mapping(
                    u'tag:yaml.org,2002:python/object:'+class_name, state)
        if isinstance(state, dict) and not state:
            return self.represent_sequence(
                    u'tag:yaml.org,2002:python/object/new:'+class_name, args)
        value = {}
        if args:
            value['args'] = args
        value['state'] = state
        return self.represent_mapping(
                u'tag:yaml.org,2002:python/object/new:'+class_name, value)

    def represent_object(self, data):
        # We use __reduce__ API to save the data. data.__reduce__ returns
        # a tuple of length 2-5:
        #   (function, args, state, listitems, dictitems)

        # For reconstructing, we calls function(*args), then set its state,
        # listitems, and dictitems if they are not None.

        # A special case is when function.__name__ == '__newobj__'. In this
        # case we create the object with args[0].__new__(*args).

        # Another special case is when __reduce__ returns a string - we don't
        # support it.

        # We produce a !!python/object, !!python/object/new or
        # !!python/object/apply node.

        cls = type(data)
        if cls in copy_reg.dispatch_table:
            reduce = copy_reg.dispatch_table[cls](data)
        elif hasattr(data, '__reduce_ex__'):
            reduce = data.__reduce_ex__(2)
        elif hasattr(data, '__reduce__'):
            reduce = data.__reduce__()
        else:
            raise RepresenterError("cannot represent an object", data)
        reduce = (list(reduce)+[None]*5)[:5]
        function, args, state, listitems, dictitems = reduce
        args = list(args)
        if state is None:
            state = {}
        if listitems is not None:
            listitems = list(listitems)
        if dictitems is not None:
            dictitems = dict(dictitems)
        if function.__name__ == '__newobj__':
            function = args[0]
            args = args[1:]
            tag = u'tag:yaml.org,2002:python/object/new:'
            newobj = True
        else:
            tag = u'tag:yaml.org,2002:python/object/apply:'
            newobj = False
        function_name = u'%s.%s' % (function.__module__, function.__name__)
        if not args and not listitems and not dictitems \
                and isinstance(state, dict) and newobj:
            return self.represent_mapping(
                    u'tag:yaml.org,2002:python/object:'+function_name, state)
        if not listitems and not dictitems  \
                and isinstance(state, dict) and not state:
            return self.represent_sequence(tag+function_name, args)
        value = {}
        if args:
            value['args'] = args
        if state or not isinstance(state, dict):
            value['state'] = state
        if listitems:
            value['listitems'] = listitems
        if dictitems:
            value['dictitems'] = dictitems
        return self.represent_mapping(tag+function_name, value)

Representer.add_representer(str,
        Representer.represent_str)

Representer.add_representer(unicode,
        Representer.represent_unicode)

Representer.add_representer(long,
        Representer.represent_long)

Representer.add_representer(complex,
        Representer.represent_complex)

Representer.add_representer(tuple,
        Representer.represent_tuple)

Representer.add_representer(type,
        Representer.represent_name)

Representer.add_representer(types.ClassType,
        Representer.represent_name)

Representer.add_representer(types.FunctionType,
        Representer.represent_name)

Representer.add_representer(types.BuiltinFunctionType,
        Representer.represent_name)

Representer.add_representer(types.ModuleType,
        Representer.represent_module)

Representer.add_multi_representer(types.InstanceType,
        Representer.represent_instance)

Representer.add_multi_representer(object,
        Representer.represent_object)

