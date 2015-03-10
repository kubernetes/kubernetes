
import yaml

def get_version_string():
    cdef char *value
    value = yaml_get_version_string()
    if PY_MAJOR_VERSION < 3:
        return value
    else:
        return PyUnicode_FromString(value)

def get_version():
    cdef int major, minor, patch
    yaml_get_version(&major, &minor, &patch)
    return (major, minor, patch)

#Mark = yaml.error.Mark
YAMLError = yaml.error.YAMLError
ReaderError = yaml.reader.ReaderError
ScannerError = yaml.scanner.ScannerError
ParserError = yaml.parser.ParserError
ComposerError = yaml.composer.ComposerError
ConstructorError = yaml.constructor.ConstructorError
EmitterError = yaml.emitter.EmitterError
SerializerError = yaml.serializer.SerializerError
RepresenterError = yaml.representer.RepresenterError

StreamStartToken = yaml.tokens.StreamStartToken
StreamEndToken = yaml.tokens.StreamEndToken
DirectiveToken = yaml.tokens.DirectiveToken
DocumentStartToken = yaml.tokens.DocumentStartToken
DocumentEndToken = yaml.tokens.DocumentEndToken
BlockSequenceStartToken = yaml.tokens.BlockSequenceStartToken
BlockMappingStartToken = yaml.tokens.BlockMappingStartToken
BlockEndToken = yaml.tokens.BlockEndToken
FlowSequenceStartToken = yaml.tokens.FlowSequenceStartToken
FlowMappingStartToken = yaml.tokens.FlowMappingStartToken
FlowSequenceEndToken = yaml.tokens.FlowSequenceEndToken
FlowMappingEndToken = yaml.tokens.FlowMappingEndToken
KeyToken = yaml.tokens.KeyToken
ValueToken = yaml.tokens.ValueToken
BlockEntryToken = yaml.tokens.BlockEntryToken
FlowEntryToken = yaml.tokens.FlowEntryToken
AliasToken = yaml.tokens.AliasToken
AnchorToken = yaml.tokens.AnchorToken
TagToken = yaml.tokens.TagToken
ScalarToken = yaml.tokens.ScalarToken

StreamStartEvent = yaml.events.StreamStartEvent
StreamEndEvent = yaml.events.StreamEndEvent
DocumentStartEvent = yaml.events.DocumentStartEvent
DocumentEndEvent = yaml.events.DocumentEndEvent
AliasEvent = yaml.events.AliasEvent
ScalarEvent = yaml.events.ScalarEvent
SequenceStartEvent = yaml.events.SequenceStartEvent
SequenceEndEvent = yaml.events.SequenceEndEvent
MappingStartEvent = yaml.events.MappingStartEvent
MappingEndEvent = yaml.events.MappingEndEvent

ScalarNode = yaml.nodes.ScalarNode
SequenceNode = yaml.nodes.SequenceNode
MappingNode = yaml.nodes.MappingNode

cdef class Mark:
    cdef readonly object name
    cdef readonly int index
    cdef readonly int line
    cdef readonly int column
    cdef readonly buffer
    cdef readonly pointer

    def __init__(self, object name, int index, int line, int column,
            object buffer, object pointer):
        self.name = name
        self.index = index
        self.line = line
        self.column = column
        self.buffer = buffer
        self.pointer = pointer

    def get_snippet(self):
        return None

    def __str__(self):
        where = "  in \"%s\", line %d, column %d"   \
                % (self.name, self.line+1, self.column+1)
        return where

#class YAMLError(Exception):
#    pass
#
#class MarkedYAMLError(YAMLError):
#
#    def __init__(self, context=None, context_mark=None,
#            problem=None, problem_mark=None, note=None):
#        self.context = context
#        self.context_mark = context_mark
#        self.problem = problem
#        self.problem_mark = problem_mark
#        self.note = note
#
#    def __str__(self):
#        lines = []
#        if self.context is not None:
#            lines.append(self.context)
#        if self.context_mark is not None  \
#            and (self.problem is None or self.problem_mark is None
#                    or self.context_mark.name != self.problem_mark.name
#                    or self.context_mark.line != self.problem_mark.line
#                    or self.context_mark.column != self.problem_mark.column):
#            lines.append(str(self.context_mark))
#        if self.problem is not None:
#            lines.append(self.problem)
#        if self.problem_mark is not None:
#            lines.append(str(self.problem_mark))
#        if self.note is not None:
#            lines.append(self.note)
#        return '\n'.join(lines)
#
#class ReaderError(YAMLError):
#
#    def __init__(self, name, position, character, encoding, reason):
#        self.name = name
#        self.character = character
#        self.position = position
#        self.encoding = encoding
#        self.reason = reason
#
#    def __str__(self):
#        if isinstance(self.character, str):
#            return "'%s' codec can't decode byte #x%02x: %s\n"  \
#                    "  in \"%s\", position %d"    \
#                    % (self.encoding, ord(self.character), self.reason,
#                            self.name, self.position)
#        else:
#            return "unacceptable character #x%04x: %s\n"    \
#                    "  in \"%s\", position %d"    \
#                    % (ord(self.character), self.reason,
#                            self.name, self.position)
#
#class ScannerError(MarkedYAMLError):
#    pass
#
#class ParserError(MarkedYAMLError):
#    pass
#
#class EmitterError(YAMLError):
#    pass
#
#cdef class Token:
#    cdef readonly Mark start_mark
#    cdef readonly Mark end_mark
#    def __init__(self, Mark start_mark, Mark end_mark):
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#
#cdef class StreamStartToken(Token):
#    cdef readonly object encoding
#    def __init__(self, Mark start_mark, Mark end_mark, encoding):
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#        self.encoding = encoding
#
#cdef class StreamEndToken(Token):
#    pass
#
#cdef class DirectiveToken(Token):
#    cdef readonly object name
#    cdef readonly object value
#    def __init__(self, name, value, Mark start_mark, Mark end_mark):
#        self.name = name
#        self.value = value
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#
#cdef class DocumentStartToken(Token):
#    pass
#
#cdef class DocumentEndToken(Token):
#    pass
#
#cdef class BlockSequenceStartToken(Token):
#    pass
#
#cdef class BlockMappingStartToken(Token):
#    pass
#
#cdef class BlockEndToken(Token):
#    pass
#
#cdef class FlowSequenceStartToken(Token):
#    pass
#
#cdef class FlowMappingStartToken(Token):
#    pass
#
#cdef class FlowSequenceEndToken(Token):
#    pass
#
#cdef class FlowMappingEndToken(Token):
#    pass
#
#cdef class KeyToken(Token):
#    pass
#
#cdef class ValueToken(Token):
#    pass
#
#cdef class BlockEntryToken(Token):
#    pass
#
#cdef class FlowEntryToken(Token):
#    pass
#
#cdef class AliasToken(Token):
#    cdef readonly object value
#    def __init__(self, value, Mark start_mark, Mark end_mark):
#        self.value = value
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#
#cdef class AnchorToken(Token):
#    cdef readonly object value
#    def __init__(self, value, Mark start_mark, Mark end_mark):
#        self.value = value
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#
#cdef class TagToken(Token):
#    cdef readonly object value
#    def __init__(self, value, Mark start_mark, Mark end_mark):
#        self.value = value
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#
#cdef class ScalarToken(Token):
#    cdef readonly object value
#    cdef readonly object plain
#    cdef readonly object style
#    def __init__(self, value, plain, Mark start_mark, Mark end_mark, style=None):
#        self.value = value
#        self.plain = plain
#        self.start_mark = start_mark
#        self.end_mark = end_mark
#        self.style = style

cdef class CParser:

    cdef yaml_parser_t parser
    cdef yaml_event_t parsed_event

    cdef object stream
    cdef object stream_name
    cdef object current_token
    cdef object current_event
    cdef object anchors
    cdef object stream_cache
    cdef int stream_cache_len
    cdef int stream_cache_pos
    cdef int unicode_source

    def __init__(self, stream):
        cdef is_readable
        if yaml_parser_initialize(&self.parser) == 0:
            raise MemoryError
        self.parsed_event.type = YAML_NO_EVENT
        is_readable = 1
        try:
            stream.read
        except AttributeError:
            is_readable = 0
        self.unicode_source = 0
        if is_readable:
            self.stream = stream
            try:
                self.stream_name = stream.name
            except AttributeError:
                if PY_MAJOR_VERSION < 3:
                    self.stream_name = '<file>'
                else:
                    self.stream_name = u'<file>'
            self.stream_cache = None
            self.stream_cache_len = 0
            self.stream_cache_pos = 0
            yaml_parser_set_input(&self.parser, input_handler, <void *>self)
        else:
            if PyUnicode_CheckExact(stream) != 0:
                stream = PyUnicode_AsUTF8String(stream)
                if PY_MAJOR_VERSION < 3:
                    self.stream_name = '<unicode string>'
                else:
                    self.stream_name = u'<unicode string>'
                self.unicode_source = 1
            else:
                if PY_MAJOR_VERSION < 3:
                    self.stream_name = '<byte string>'
                else:
                    self.stream_name = u'<byte string>'
            if PyString_CheckExact(stream) == 0:
                if PY_MAJOR_VERSION < 3:
                    raise TypeError("a string or stream input is required")
                else:
                    raise TypeError(u"a string or stream input is required")
            self.stream = stream
            yaml_parser_set_input_string(&self.parser, PyString_AS_STRING(stream), PyString_GET_SIZE(stream))
        self.current_token = None
        self.current_event = None
        self.anchors = {}

    def __dealloc__(self):
        yaml_parser_delete(&self.parser)
        yaml_event_delete(&self.parsed_event)

    def dispose(self):
        pass

    cdef object _parser_error(self):
        if self.parser.error == YAML_MEMORY_ERROR:
            return MemoryError
        elif self.parser.error == YAML_READER_ERROR:
            if PY_MAJOR_VERSION < 3:
                return ReaderError(self.stream_name, self.parser.problem_offset,
                        self.parser.problem_value, '?', self.parser.problem)
            else:
                return ReaderError(self.stream_name, self.parser.problem_offset,
                        self.parser.problem_value, u'?', PyUnicode_FromString(self.parser.problem))
        elif self.parser.error == YAML_SCANNER_ERROR    \
                or self.parser.error == YAML_PARSER_ERROR:
            context_mark = None
            problem_mark = None
            if self.parser.context != NULL:
                context_mark = Mark(self.stream_name,
                        self.parser.context_mark.index,
                        self.parser.context_mark.line,
                        self.parser.context_mark.column, None, None)
            if self.parser.problem != NULL:
                problem_mark = Mark(self.stream_name,
                        self.parser.problem_mark.index,
                        self.parser.problem_mark.line,
                        self.parser.problem_mark.column, None, None)
            context = None
            if self.parser.context != NULL:
                if PY_MAJOR_VERSION < 3:
                    context = self.parser.context
                else:
                    context = PyUnicode_FromString(self.parser.context)
            if PY_MAJOR_VERSION < 3:
                problem = self.parser.problem
            else:
                problem = PyUnicode_FromString(self.parser.problem)
            if self.parser.error == YAML_SCANNER_ERROR:
                return ScannerError(context, context_mark, problem, problem_mark)
            else:
                return ParserError(context, context_mark, problem, problem_mark)
        if PY_MAJOR_VERSION < 3:
            raise ValueError("no parser error")
        else:
            raise ValueError(u"no parser error")

    def raw_scan(self):
        cdef yaml_token_t token
        cdef int done
        cdef int count
        count = 0
        done = 0
        while done == 0:
            if yaml_parser_scan(&self.parser, &token) == 0:
                error = self._parser_error()
                raise error
            if token.type == YAML_NO_TOKEN:
                done = 1
            else:
                count = count+1
            yaml_token_delete(&token)
        return count

    cdef object _scan(self):
        cdef yaml_token_t token
        if yaml_parser_scan(&self.parser, &token) == 0:
            error = self._parser_error()
            raise error
        token_object = self._token_to_object(&token)
        yaml_token_delete(&token)
        return token_object

    cdef object _token_to_object(self, yaml_token_t *token):
        start_mark = Mark(self.stream_name,
                token.start_mark.index,
                token.start_mark.line,
                token.start_mark.column,
                None, None)
        end_mark = Mark(self.stream_name,
                token.end_mark.index,
                token.end_mark.line,
                token.end_mark.column,
                None, None)
        if token.type == YAML_NO_TOKEN:
            return None
        elif token.type == YAML_STREAM_START_TOKEN:
            encoding = None
            if token.data.stream_start.encoding == YAML_UTF8_ENCODING:
                if self.unicode_source == 0:
                    encoding = u"utf-8"
            elif token.data.stream_start.encoding == YAML_UTF16LE_ENCODING:
                encoding = u"utf-16-le"
            elif token.data.stream_start.encoding == YAML_UTF16BE_ENCODING:
                encoding = u"utf-16-be"
            return StreamStartToken(start_mark, end_mark, encoding)
        elif token.type == YAML_STREAM_END_TOKEN:
            return StreamEndToken(start_mark, end_mark)
        elif token.type == YAML_VERSION_DIRECTIVE_TOKEN:
            return DirectiveToken(u"YAML",
                    (token.data.version_directive.major,
                        token.data.version_directive.minor),
                    start_mark, end_mark)
        elif token.type == YAML_TAG_DIRECTIVE_TOKEN:
            handle = PyUnicode_FromString(token.data.tag_directive.handle)
            prefix = PyUnicode_FromString(token.data.tag_directive.prefix)
            return DirectiveToken(u"TAG", (handle, prefix),
                    start_mark, end_mark)
        elif token.type == YAML_DOCUMENT_START_TOKEN:
            return DocumentStartToken(start_mark, end_mark)
        elif token.type == YAML_DOCUMENT_END_TOKEN:
            return DocumentEndToken(start_mark, end_mark)
        elif token.type == YAML_BLOCK_SEQUENCE_START_TOKEN:
            return BlockSequenceStartToken(start_mark, end_mark)
        elif token.type == YAML_BLOCK_MAPPING_START_TOKEN:
            return BlockMappingStartToken(start_mark, end_mark)
        elif token.type == YAML_BLOCK_END_TOKEN:
            return BlockEndToken(start_mark, end_mark)
        elif token.type == YAML_FLOW_SEQUENCE_START_TOKEN:
            return FlowSequenceStartToken(start_mark, end_mark)
        elif token.type == YAML_FLOW_SEQUENCE_END_TOKEN:
            return FlowSequenceEndToken(start_mark, end_mark)
        elif token.type == YAML_FLOW_MAPPING_START_TOKEN:
            return FlowMappingStartToken(start_mark, end_mark)
        elif token.type == YAML_FLOW_MAPPING_END_TOKEN:
            return FlowMappingEndToken(start_mark, end_mark)
        elif token.type == YAML_BLOCK_ENTRY_TOKEN:
            return BlockEntryToken(start_mark, end_mark)
        elif token.type == YAML_FLOW_ENTRY_TOKEN:
            return FlowEntryToken(start_mark, end_mark)
        elif token.type == YAML_KEY_TOKEN:
            return KeyToken(start_mark, end_mark)
        elif token.type == YAML_VALUE_TOKEN:
            return ValueToken(start_mark, end_mark)
        elif token.type == YAML_ALIAS_TOKEN:
            value = PyUnicode_FromString(token.data.alias.value)
            return AliasToken(value, start_mark, end_mark)
        elif token.type == YAML_ANCHOR_TOKEN:
            value = PyUnicode_FromString(token.data.anchor.value)
            return AnchorToken(value, start_mark, end_mark)
        elif token.type == YAML_TAG_TOKEN:
            handle = PyUnicode_FromString(token.data.tag.handle)
            suffix = PyUnicode_FromString(token.data.tag.suffix)
            if not handle:
                handle = None
            return TagToken((handle, suffix), start_mark, end_mark)
        elif token.type == YAML_SCALAR_TOKEN:
            value = PyUnicode_DecodeUTF8(token.data.scalar.value,
                    token.data.scalar.length, 'strict')
            plain = False
            style = None
            if token.data.scalar.style == YAML_PLAIN_SCALAR_STYLE:
                plain = True
                style = u''
            elif token.data.scalar.style == YAML_SINGLE_QUOTED_SCALAR_STYLE:
                style = u'\''
            elif token.data.scalar.style == YAML_DOUBLE_QUOTED_SCALAR_STYLE:
                style = u'"'
            elif token.data.scalar.style == YAML_LITERAL_SCALAR_STYLE:
                style = u'|'
            elif token.data.scalar.style == YAML_FOLDED_SCALAR_STYLE:
                style = u'>'
            return ScalarToken(value, plain,
                    start_mark, end_mark, style)
        else:
            if PY_MAJOR_VERSION < 3:
                raise ValueError("unknown token type")
            else:
                raise ValueError(u"unknown token type")

    def get_token(self):
        if self.current_token is not None:
            value = self.current_token
            self.current_token = None
        else:
            value = self._scan()
        return value

    def peek_token(self):
        if self.current_token is None:
            self.current_token = self._scan()
        return self.current_token

    def check_token(self, *choices):
        if self.current_token is None:
            self.current_token = self._scan()
        if self.current_token is None:
            return False
        if not choices:
            return True
        token_class = self.current_token.__class__
        for choice in choices:
            if token_class is choice:
                return True
        return False

    def raw_parse(self):
        cdef yaml_event_t event
        cdef int done
        cdef int count
        count = 0
        done = 0
        while done == 0:
            if yaml_parser_parse(&self.parser, &event) == 0:
                error = self._parser_error()
                raise error
            if event.type == YAML_NO_EVENT:
                done = 1
            else:
                count = count+1
            yaml_event_delete(&event)
        return count

    cdef object _parse(self):
        cdef yaml_event_t event
        if yaml_parser_parse(&self.parser, &event) == 0:
            error = self._parser_error()
            raise error
        event_object = self._event_to_object(&event)
        yaml_event_delete(&event)
        return event_object

    cdef object _event_to_object(self, yaml_event_t *event):
        cdef yaml_tag_directive_t *tag_directive
        start_mark = Mark(self.stream_name,
                event.start_mark.index,
                event.start_mark.line,
                event.start_mark.column,
                None, None)
        end_mark = Mark(self.stream_name,
                event.end_mark.index,
                event.end_mark.line,
                event.end_mark.column,
                None, None)
        if event.type == YAML_NO_EVENT:
            return None
        elif event.type == YAML_STREAM_START_EVENT:
            encoding = None
            if event.data.stream_start.encoding == YAML_UTF8_ENCODING:
                if self.unicode_source == 0:
                    encoding = u"utf-8"
            elif event.data.stream_start.encoding == YAML_UTF16LE_ENCODING:
                encoding = u"utf-16-le"
            elif event.data.stream_start.encoding == YAML_UTF16BE_ENCODING:
                encoding = u"utf-16-be"
            return StreamStartEvent(start_mark, end_mark, encoding)
        elif event.type == YAML_STREAM_END_EVENT:
            return StreamEndEvent(start_mark, end_mark)
        elif event.type == YAML_DOCUMENT_START_EVENT:
            explicit = False
            if event.data.document_start.implicit == 0:
                explicit = True
            version = None
            if event.data.document_start.version_directive != NULL:
                version = (event.data.document_start.version_directive.major,
                        event.data.document_start.version_directive.minor)
            tags = None
            if event.data.document_start.tag_directives.start != NULL:
                tags = {}
                tag_directive = event.data.document_start.tag_directives.start
                while tag_directive != event.data.document_start.tag_directives.end:
                    handle = PyUnicode_FromString(tag_directive.handle)
                    prefix = PyUnicode_FromString(tag_directive.prefix)
                    tags[handle] = prefix
                    tag_directive = tag_directive+1
            return DocumentStartEvent(start_mark, end_mark,
                    explicit, version, tags)
        elif event.type == YAML_DOCUMENT_END_EVENT:
            explicit = False
            if event.data.document_end.implicit == 0:
                explicit = True
            return DocumentEndEvent(start_mark, end_mark, explicit)
        elif event.type == YAML_ALIAS_EVENT:
            anchor = PyUnicode_FromString(event.data.alias.anchor)
            return AliasEvent(anchor, start_mark, end_mark)
        elif event.type == YAML_SCALAR_EVENT:
            anchor = None
            if event.data.scalar.anchor != NULL:
                anchor = PyUnicode_FromString(event.data.scalar.anchor)
            tag = None
            if event.data.scalar.tag != NULL:
                tag = PyUnicode_FromString(event.data.scalar.tag)
            value = PyUnicode_DecodeUTF8(event.data.scalar.value,
                    event.data.scalar.length, 'strict')
            plain_implicit = False
            if event.data.scalar.plain_implicit == 1:
                plain_implicit = True
            quoted_implicit = False
            if event.data.scalar.quoted_implicit == 1:
                quoted_implicit = True
            style = None
            if event.data.scalar.style == YAML_PLAIN_SCALAR_STYLE:
                style = u''
            elif event.data.scalar.style == YAML_SINGLE_QUOTED_SCALAR_STYLE:
                style = u'\''
            elif event.data.scalar.style == YAML_DOUBLE_QUOTED_SCALAR_STYLE:
                style = u'"'
            elif event.data.scalar.style == YAML_LITERAL_SCALAR_STYLE:
                style = u'|'
            elif event.data.scalar.style == YAML_FOLDED_SCALAR_STYLE:
                style = u'>'
            return ScalarEvent(anchor, tag,
                    (plain_implicit, quoted_implicit),
                    value, start_mark, end_mark, style)
        elif event.type == YAML_SEQUENCE_START_EVENT:
            anchor = None
            if event.data.sequence_start.anchor != NULL:
                anchor = PyUnicode_FromString(event.data.sequence_start.anchor)
            tag = None
            if event.data.sequence_start.tag != NULL:
                tag = PyUnicode_FromString(event.data.sequence_start.tag)
            implicit = False
            if event.data.sequence_start.implicit == 1:
                implicit = True
            flow_style = None
            if event.data.sequence_start.style == YAML_FLOW_SEQUENCE_STYLE:
                flow_style = True
            elif event.data.sequence_start.style == YAML_BLOCK_SEQUENCE_STYLE:
                flow_style = False
            return SequenceStartEvent(anchor, tag, implicit,
                    start_mark, end_mark, flow_style)
        elif event.type == YAML_MAPPING_START_EVENT:
            anchor = None
            if event.data.mapping_start.anchor != NULL:
                anchor = PyUnicode_FromString(event.data.mapping_start.anchor)
            tag = None
            if event.data.mapping_start.tag != NULL:
                tag = PyUnicode_FromString(event.data.mapping_start.tag)
            implicit = False
            if event.data.mapping_start.implicit == 1:
                implicit = True
            flow_style = None
            if event.data.mapping_start.style == YAML_FLOW_MAPPING_STYLE:
                flow_style = True
            elif event.data.mapping_start.style == YAML_BLOCK_MAPPING_STYLE:
                flow_style = False
            return MappingStartEvent(anchor, tag, implicit,
                    start_mark, end_mark, flow_style)
        elif event.type == YAML_SEQUENCE_END_EVENT:
            return SequenceEndEvent(start_mark, end_mark)
        elif event.type == YAML_MAPPING_END_EVENT:
            return MappingEndEvent(start_mark, end_mark)
        else:
            if PY_MAJOR_VERSION < 3:
                raise ValueError("unknown event type")
            else:
                raise ValueError(u"unknown event type")

    def get_event(self):
        if self.current_event is not None:
            value = self.current_event
            self.current_event = None
        else:
            value = self._parse()
        return value

    def peek_event(self):
        if self.current_event is None:
            self.current_event = self._parse()
        return self.current_event

    def check_event(self, *choices):
        if self.current_event is None:
            self.current_event = self._parse()
        if self.current_event is None:
            return False
        if not choices:
            return True
        event_class = self.current_event.__class__
        for choice in choices:
            if event_class is choice:
                return True
        return False

    def check_node(self):
        self._parse_next_event()
        if self.parsed_event.type == YAML_STREAM_START_EVENT:
            yaml_event_delete(&self.parsed_event)
            self._parse_next_event()
        if self.parsed_event.type != YAML_STREAM_END_EVENT:
            return True
        return False

    def get_node(self):
        self._parse_next_event()
        if self.parsed_event.type != YAML_STREAM_END_EVENT:
            return self._compose_document()

    def get_single_node(self):
        self._parse_next_event()
        yaml_event_delete(&self.parsed_event)
        self._parse_next_event()
        document = None
        if self.parsed_event.type != YAML_STREAM_END_EVENT:
            document = self._compose_document()
        self._parse_next_event()
        if self.parsed_event.type != YAML_STREAM_END_EVENT:
            mark = Mark(self.stream_name,
                    self.parsed_event.start_mark.index,
                    self.parsed_event.start_mark.line,
                    self.parsed_event.start_mark.column,
                    None, None)
            if PY_MAJOR_VERSION < 3:
                raise ComposerError("expected a single document in the stream",
                        document.start_mark, "but found another document", mark)
            else:
                raise ComposerError(u"expected a single document in the stream",
                        document.start_mark, u"but found another document", mark)
        return document

    cdef object _compose_document(self):
        yaml_event_delete(&self.parsed_event)
        node = self._compose_node(None, None)
        self._parse_next_event()
        yaml_event_delete(&self.parsed_event)
        self.anchors = {}
        return node

    cdef object _compose_node(self, object parent, object index):
        self._parse_next_event()
        if self.parsed_event.type == YAML_ALIAS_EVENT:
            anchor = PyUnicode_FromString(self.parsed_event.data.alias.anchor)
            if anchor not in self.anchors:
                mark = Mark(self.stream_name,
                        self.parsed_event.start_mark.index,
                        self.parsed_event.start_mark.line,
                        self.parsed_event.start_mark.column,
                        None, None)
                if PY_MAJOR_VERSION < 3:
                    raise ComposerError(None, None, "found undefined alias", mark)
                else:
                    raise ComposerError(None, None, u"found undefined alias", mark)
            yaml_event_delete(&self.parsed_event)
            return self.anchors[anchor]
        anchor = None
        if self.parsed_event.type == YAML_SCALAR_EVENT  \
                and self.parsed_event.data.scalar.anchor != NULL:
            anchor = PyUnicode_FromString(self.parsed_event.data.scalar.anchor)
        elif self.parsed_event.type == YAML_SEQUENCE_START_EVENT    \
                and self.parsed_event.data.sequence_start.anchor != NULL:
            anchor = PyUnicode_FromString(self.parsed_event.data.sequence_start.anchor)
        elif self.parsed_event.type == YAML_MAPPING_START_EVENT    \
                and self.parsed_event.data.mapping_start.anchor != NULL:
            anchor = PyUnicode_FromString(self.parsed_event.data.mapping_start.anchor)
        if anchor is not None:
            if anchor in self.anchors:
                mark = Mark(self.stream_name,
                        self.parsed_event.start_mark.index,
                        self.parsed_event.start_mark.line,
                        self.parsed_event.start_mark.column,
                        None, None)
                if PY_MAJOR_VERSION < 3:
                    raise ComposerError("found duplicate anchor; first occurence",
                            self.anchors[anchor].start_mark, "second occurence", mark)
                else:
                    raise ComposerError(u"found duplicate anchor; first occurence",
                            self.anchors[anchor].start_mark, u"second occurence", mark)
        self.descend_resolver(parent, index)
        if self.parsed_event.type == YAML_SCALAR_EVENT:
            node = self._compose_scalar_node(anchor)
        elif self.parsed_event.type == YAML_SEQUENCE_START_EVENT:
            node = self._compose_sequence_node(anchor)
        elif self.parsed_event.type == YAML_MAPPING_START_EVENT:
            node = self._compose_mapping_node(anchor)
        self.ascend_resolver()
        return node

    cdef _compose_scalar_node(self, object anchor):
        start_mark = Mark(self.stream_name,
                self.parsed_event.start_mark.index,
                self.parsed_event.start_mark.line,
                self.parsed_event.start_mark.column,
                None, None)
        end_mark = Mark(self.stream_name,
                self.parsed_event.end_mark.index,
                self.parsed_event.end_mark.line,
                self.parsed_event.end_mark.column,
                None, None)
        value = PyUnicode_DecodeUTF8(self.parsed_event.data.scalar.value,
                self.parsed_event.data.scalar.length, 'strict')
        plain_implicit = False
        if self.parsed_event.data.scalar.plain_implicit == 1:
            plain_implicit = True
        quoted_implicit = False
        if self.parsed_event.data.scalar.quoted_implicit == 1:
            quoted_implicit = True
        if self.parsed_event.data.scalar.tag == NULL    \
                or (self.parsed_event.data.scalar.tag[0] == c'!'
                        and self.parsed_event.data.scalar.tag[1] == c'\0'):
            tag = self.resolve(ScalarNode, value, (plain_implicit, quoted_implicit))
        else:
            tag = PyUnicode_FromString(self.parsed_event.data.scalar.tag)
        style = None
        if self.parsed_event.data.scalar.style == YAML_PLAIN_SCALAR_STYLE:
            style = u''
        elif self.parsed_event.data.scalar.style == YAML_SINGLE_QUOTED_SCALAR_STYLE:
            style = u'\''
        elif self.parsed_event.data.scalar.style == YAML_DOUBLE_QUOTED_SCALAR_STYLE:
            style = u'"'
        elif self.parsed_event.data.scalar.style == YAML_LITERAL_SCALAR_STYLE:
            style = u'|'
        elif self.parsed_event.data.scalar.style == YAML_FOLDED_SCALAR_STYLE:
            style = u'>'
        node = ScalarNode(tag, value, start_mark, end_mark, style)
        if anchor is not None:
            self.anchors[anchor] = node
        yaml_event_delete(&self.parsed_event)
        return node

    cdef _compose_sequence_node(self, object anchor):
        cdef int index
        start_mark = Mark(self.stream_name,
                self.parsed_event.start_mark.index,
                self.parsed_event.start_mark.line,
                self.parsed_event.start_mark.column,
                None, None)
        implicit = False
        if self.parsed_event.data.sequence_start.implicit == 1:
            implicit = True
        if self.parsed_event.data.sequence_start.tag == NULL    \
                or (self.parsed_event.data.sequence_start.tag[0] == c'!'
                        and self.parsed_event.data.sequence_start.tag[1] == c'\0'):
            tag = self.resolve(SequenceNode, None, implicit)
        else:
            tag = PyUnicode_FromString(self.parsed_event.data.sequence_start.tag)
        flow_style = None
        if self.parsed_event.data.sequence_start.style == YAML_FLOW_SEQUENCE_STYLE:
            flow_style = True
        elif self.parsed_event.data.sequence_start.style == YAML_BLOCK_SEQUENCE_STYLE:
            flow_style = False
        value = []
        node = SequenceNode(tag, value, start_mark, None, flow_style)
        if anchor is not None:
            self.anchors[anchor] = node
        yaml_event_delete(&self.parsed_event)
        index = 0
        self._parse_next_event()
        while self.parsed_event.type != YAML_SEQUENCE_END_EVENT:
            value.append(self._compose_node(node, index))
            index = index+1
            self._parse_next_event()
        node.end_mark = Mark(self.stream_name,
                self.parsed_event.end_mark.index,
                self.parsed_event.end_mark.line,
                self.parsed_event.end_mark.column,
                None, None)
        yaml_event_delete(&self.parsed_event)
        return node

    cdef _compose_mapping_node(self, object anchor):
        start_mark = Mark(self.stream_name,
                self.parsed_event.start_mark.index,
                self.parsed_event.start_mark.line,
                self.parsed_event.start_mark.column,
                None, None)
        implicit = False
        if self.parsed_event.data.mapping_start.implicit == 1:
            implicit = True
        if self.parsed_event.data.mapping_start.tag == NULL    \
                or (self.parsed_event.data.mapping_start.tag[0] == c'!'
                        and self.parsed_event.data.mapping_start.tag[1] == c'\0'):
            tag = self.resolve(MappingNode, None, implicit)
        else:
            tag = PyUnicode_FromString(self.parsed_event.data.mapping_start.tag)
        flow_style = None
        if self.parsed_event.data.mapping_start.style == YAML_FLOW_MAPPING_STYLE:
            flow_style = True
        elif self.parsed_event.data.mapping_start.style == YAML_BLOCK_MAPPING_STYLE:
            flow_style = False
        value = []
        node = MappingNode(tag, value, start_mark, None, flow_style)
        if anchor is not None:
            self.anchors[anchor] = node
        yaml_event_delete(&self.parsed_event)
        self._parse_next_event()
        while self.parsed_event.type != YAML_MAPPING_END_EVENT:
            item_key = self._compose_node(node, None)
            item_value = self._compose_node(node, item_key)
            value.append((item_key, item_value))
            self._parse_next_event()
        node.end_mark = Mark(self.stream_name,
                self.parsed_event.end_mark.index,
                self.parsed_event.end_mark.line,
                self.parsed_event.end_mark.column,
                None, None)
        yaml_event_delete(&self.parsed_event)
        return node

    cdef int _parse_next_event(self) except 0:
        if self.parsed_event.type == YAML_NO_EVENT:
            if yaml_parser_parse(&self.parser, &self.parsed_event) == 0:
                error = self._parser_error()
                raise error
        return 1

cdef int input_handler(void *data, char *buffer, int size, int *read) except 0:
    cdef CParser parser
    parser = <CParser>data
    if parser.stream_cache is None:
        value = parser.stream.read(size)
        if PyUnicode_CheckExact(value) != 0:
            value = PyUnicode_AsUTF8String(value)
            parser.unicode_source = 1
        if PyString_CheckExact(value) == 0:
            if PY_MAJOR_VERSION < 3:
                raise TypeError("a string value is expected")
            else:
                raise TypeError(u"a string value is expected")
        parser.stream_cache = value
        parser.stream_cache_pos = 0
        parser.stream_cache_len = PyString_GET_SIZE(value)
    if (parser.stream_cache_len - parser.stream_cache_pos) < size:
        size = parser.stream_cache_len - parser.stream_cache_pos
    if size > 0:
        memcpy(buffer, PyString_AS_STRING(parser.stream_cache)
                            + parser.stream_cache_pos, size)
    read[0] = size
    parser.stream_cache_pos += size
    if parser.stream_cache_pos == parser.stream_cache_len:
        parser.stream_cache = None
    return 1

cdef class CEmitter:

    cdef yaml_emitter_t emitter

    cdef object stream

    cdef int document_start_implicit
    cdef int document_end_implicit
    cdef object use_version
    cdef object use_tags

    cdef object serialized_nodes
    cdef object anchors
    cdef int last_alias_id
    cdef int closed
    cdef int dump_unicode
    cdef object use_encoding

    def __init__(self, stream, canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None, encoding=None,
            explicit_start=None, explicit_end=None, version=None, tags=None):
        if yaml_emitter_initialize(&self.emitter) == 0:
            raise MemoryError
        self.stream = stream
        self.dump_unicode = 0
        if PY_MAJOR_VERSION < 3:
            if getattr3(stream, 'encoding', None):
                self.dump_unicode = 1
        else:
            if hasattr(stream, u'encoding'):
                self.dump_unicode = 1
        self.use_encoding = encoding
        yaml_emitter_set_output(&self.emitter, output_handler, <void *>self)    
        if canonical:
            yaml_emitter_set_canonical(&self.emitter, 1)
        if indent is not None:
            yaml_emitter_set_indent(&self.emitter, indent)
        if width is not None:
            yaml_emitter_set_width(&self.emitter, width)
        if allow_unicode:
            yaml_emitter_set_unicode(&self.emitter, 1)
        if line_break is not None:
            if line_break == '\r':
                yaml_emitter_set_break(&self.emitter, YAML_CR_BREAK)
            elif line_break == '\n':
                yaml_emitter_set_break(&self.emitter, YAML_LN_BREAK)
            elif line_break == '\r\n':
                yaml_emitter_set_break(&self.emitter, YAML_CRLN_BREAK)
        self.document_start_implicit = 1
        if explicit_start:
            self.document_start_implicit = 0
        self.document_end_implicit = 1
        if explicit_end:
            self.document_end_implicit = 0
        self.use_version = version
        self.use_tags = tags
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_alias_id = 0
        self.closed = -1

    def __dealloc__(self):
        yaml_emitter_delete(&self.emitter)

    def dispose(self):
        pass

    cdef object _emitter_error(self):
        if self.emitter.error == YAML_MEMORY_ERROR:
            return MemoryError
        elif self.emitter.error == YAML_EMITTER_ERROR:
            if PY_MAJOR_VERSION < 3:
                problem = self.emitter.problem
            else:
                problem = PyUnicode_FromString(self.emitter.problem)
            return EmitterError(problem)
        if PY_MAJOR_VERSION < 3:
            raise ValueError("no emitter error")
        else:
            raise ValueError(u"no emitter error")

    cdef int _object_to_event(self, object event_object, yaml_event_t *event) except 0:
        cdef yaml_encoding_t encoding
        cdef yaml_version_directive_t version_directive_value
        cdef yaml_version_directive_t *version_directive
        cdef yaml_tag_directive_t tag_directives_value[128]
        cdef yaml_tag_directive_t *tag_directives_start
        cdef yaml_tag_directive_t *tag_directives_end
        cdef int implicit
        cdef int plain_implicit
        cdef int quoted_implicit
        cdef char *anchor
        cdef char *tag
        cdef char *value
        cdef int length
        cdef yaml_scalar_style_t scalar_style
        cdef yaml_sequence_style_t sequence_style
        cdef yaml_mapping_style_t mapping_style
        event_class = event_object.__class__
        if event_class is StreamStartEvent:
            encoding = YAML_UTF8_ENCODING
            if event_object.encoding == u'utf-16-le' or event_object.encoding == 'utf-16-le':
                encoding = YAML_UTF16LE_ENCODING
            elif event_object.encoding == u'utf-16-be' or event_object.encoding == 'utf-16-be':
                encoding = YAML_UTF16BE_ENCODING
            if event_object.encoding is None:
                self.dump_unicode = 1
            if self.dump_unicode == 1:
                encoding = YAML_UTF8_ENCODING
            yaml_stream_start_event_initialize(event, encoding)
        elif event_class is StreamEndEvent:
            yaml_stream_end_event_initialize(event)
        elif event_class is DocumentStartEvent:
            version_directive = NULL
            if event_object.version:
                version_directive_value.major = event_object.version[0]
                version_directive_value.minor = event_object.version[1]
                version_directive = &version_directive_value
            tag_directives_start = NULL
            tag_directives_end = NULL
            if event_object.tags:
                if len(event_object.tags) > 128:
                    if PY_MAJOR_VERSION < 3:
                        raise ValueError("too many tags")
                    else:
                        raise ValueError(u"too many tags")
                tag_directives_start = tag_directives_value
                tag_directives_end = tag_directives_value
                cache = []
                for handle in event_object.tags:
                    prefix = event_object.tags[handle]
                    if PyUnicode_CheckExact(handle):
                        handle = PyUnicode_AsUTF8String(handle)
                        cache.append(handle)
                    if not PyString_CheckExact(handle):
                        if PY_MAJOR_VERSION < 3:
                            raise TypeError("tag handle must be a string")
                        else:
                            raise TypeError(u"tag handle must be a string")
                    tag_directives_end.handle = PyString_AS_STRING(handle)
                    if PyUnicode_CheckExact(prefix):
                        prefix = PyUnicode_AsUTF8String(prefix)
                        cache.append(prefix)
                    if not PyString_CheckExact(prefix):
                        if PY_MAJOR_VERSION < 3:
                            raise TypeError("tag prefix must be a string")
                        else:
                            raise TypeError(u"tag prefix must be a string")
                    tag_directives_end.prefix = PyString_AS_STRING(prefix)
                    tag_directives_end = tag_directives_end+1
            implicit = 1
            if event_object.explicit:
                implicit = 0
            if yaml_document_start_event_initialize(event, version_directive,
                    tag_directives_start, tag_directives_end, implicit) == 0:
                raise MemoryError
        elif event_class is DocumentEndEvent:
            implicit = 1
            if event_object.explicit:
                implicit = 0
            yaml_document_end_event_initialize(event, implicit)
        elif event_class is AliasEvent:
            anchor = NULL
            anchor_object = event_object.anchor
            if PyUnicode_CheckExact(anchor_object):
                anchor_object = PyUnicode_AsUTF8String(anchor_object)
            if not PyString_CheckExact(anchor_object):
                if PY_MAJOR_VERSION < 3:
                    raise TypeError("anchor must be a string")
                else:
                    raise TypeError(u"anchor must be a string")
            anchor = PyString_AS_STRING(anchor_object)
            if yaml_alias_event_initialize(event, anchor) == 0:
                raise MemoryError
        elif event_class is ScalarEvent:
            anchor = NULL
            anchor_object = event_object.anchor
            if anchor_object is not None:
                if PyUnicode_CheckExact(anchor_object):
                    anchor_object = PyUnicode_AsUTF8String(anchor_object)
                if not PyString_CheckExact(anchor_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("anchor must be a string")
                    else:
                        raise TypeError(u"anchor must be a string")
                anchor = PyString_AS_STRING(anchor_object)
            tag = NULL
            tag_object = event_object.tag
            if tag_object is not None:
                if PyUnicode_CheckExact(tag_object):
                    tag_object = PyUnicode_AsUTF8String(tag_object)
                if not PyString_CheckExact(tag_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("tag must be a string")
                    else:
                        raise TypeError(u"tag must be a string")
                tag = PyString_AS_STRING(tag_object)
            value_object = event_object.value
            if PyUnicode_CheckExact(value_object):
                value_object = PyUnicode_AsUTF8String(value_object)
            if not PyString_CheckExact(value_object):
                if PY_MAJOR_VERSION < 3:
                    raise TypeError("value must be a string")
                else:
                    raise TypeError(u"value must be a string")
            value = PyString_AS_STRING(value_object)
            length = PyString_GET_SIZE(value_object)
            plain_implicit = 0
            quoted_implicit = 0
            if event_object.implicit is not None:
                plain_implicit = event_object.implicit[0]
                quoted_implicit = event_object.implicit[1]
            style_object = event_object.style
            scalar_style = YAML_PLAIN_SCALAR_STYLE
            if style_object == "'" or style_object == u"'":
                scalar_style = YAML_SINGLE_QUOTED_SCALAR_STYLE
            elif style_object == "\"" or style_object == u"\"":
                scalar_style = YAML_DOUBLE_QUOTED_SCALAR_STYLE
            elif style_object == "|" or style_object == u"|":
                scalar_style = YAML_LITERAL_SCALAR_STYLE
            elif style_object == ">" or style_object == u">":
                scalar_style = YAML_FOLDED_SCALAR_STYLE
            if yaml_scalar_event_initialize(event, anchor, tag, value, length,
                    plain_implicit, quoted_implicit, scalar_style) == 0:
                raise MemoryError
        elif event_class is SequenceStartEvent:
            anchor = NULL
            anchor_object = event_object.anchor
            if anchor_object is not None:
                if PyUnicode_CheckExact(anchor_object):
                    anchor_object = PyUnicode_AsUTF8String(anchor_object)
                if not PyString_CheckExact(anchor_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("anchor must be a string")
                    else:
                        raise TypeError(u"anchor must be a string")
                anchor = PyString_AS_STRING(anchor_object)
            tag = NULL
            tag_object = event_object.tag
            if tag_object is not None:
                if PyUnicode_CheckExact(tag_object):
                    tag_object = PyUnicode_AsUTF8String(tag_object)
                if not PyString_CheckExact(tag_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("tag must be a string")
                    else:
                        raise TypeError(u"tag must be a string")
                tag = PyString_AS_STRING(tag_object)
            implicit = 0
            if event_object.implicit:
                implicit = 1
            sequence_style = YAML_BLOCK_SEQUENCE_STYLE
            if event_object.flow_style:
                sequence_style = YAML_FLOW_SEQUENCE_STYLE
            if yaml_sequence_start_event_initialize(event, anchor, tag,
                    implicit, sequence_style) == 0:
                raise MemoryError
        elif event_class is MappingStartEvent:
            anchor = NULL
            anchor_object = event_object.anchor
            if anchor_object is not None:
                if PyUnicode_CheckExact(anchor_object):
                    anchor_object = PyUnicode_AsUTF8String(anchor_object)
                if not PyString_CheckExact(anchor_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("anchor must be a string")
                    else:
                        raise TypeError(u"anchor must be a string")
                anchor = PyString_AS_STRING(anchor_object)
            tag = NULL
            tag_object = event_object.tag
            if tag_object is not None:
                if PyUnicode_CheckExact(tag_object):
                    tag_object = PyUnicode_AsUTF8String(tag_object)
                if not PyString_CheckExact(tag_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("tag must be a string")
                    else:
                        raise TypeError(u"tag must be a string")
                tag = PyString_AS_STRING(tag_object)
            implicit = 0
            if event_object.implicit:
                implicit = 1
            mapping_style = YAML_BLOCK_MAPPING_STYLE
            if event_object.flow_style:
                mapping_style = YAML_FLOW_MAPPING_STYLE
            if yaml_mapping_start_event_initialize(event, anchor, tag,
                    implicit, mapping_style) == 0:
                raise MemoryError
        elif event_class is SequenceEndEvent:
            yaml_sequence_end_event_initialize(event)
        elif event_class is MappingEndEvent:
            yaml_mapping_end_event_initialize(event)
        else:
            if PY_MAJOR_VERSION < 3:
                raise TypeError("invalid event %s" % event_object)
            else:
                raise TypeError(u"invalid event %s" % event_object)
        return 1

    def emit(self, event_object):
        cdef yaml_event_t event
        self._object_to_event(event_object, &event)
        if yaml_emitter_emit(&self.emitter, &event) == 0:
            error = self._emitter_error()
            raise error

    def open(self):
        cdef yaml_event_t event
        cdef yaml_encoding_t encoding
        if self.closed == -1:
            if self.use_encoding == u'utf-16-le' or self.use_encoding == 'utf-16-le':
                encoding = YAML_UTF16LE_ENCODING
            elif self.use_encoding == u'utf-16-be' or self.use_encoding == 'utf-16-be':
                encoding = YAML_UTF16BE_ENCODING
            else:
                encoding = YAML_UTF8_ENCODING
            if self.use_encoding is None:
                self.dump_unicode = 1
            if self.dump_unicode == 1:
                encoding = YAML_UTF8_ENCODING
            yaml_stream_start_event_initialize(&event, encoding)
            if yaml_emitter_emit(&self.emitter, &event) == 0:
                error = self._emitter_error()
                raise error
            self.closed = 0
        elif self.closed == 1:
            if PY_MAJOR_VERSION < 3:
                raise SerializerError("serializer is closed")
            else:
                raise SerializerError(u"serializer is closed")
        else:
            if PY_MAJOR_VERSION < 3:
                raise SerializerError("serializer is already opened")
            else:
                raise SerializerError(u"serializer is already opened")

    def close(self):
        cdef yaml_event_t event
        if self.closed == -1:
            if PY_MAJOR_VERSION < 3:
                raise SerializerError("serializer is not opened")
            else:
                raise SerializerError(u"serializer is not opened")
        elif self.closed == 0:
            yaml_stream_end_event_initialize(&event)
            if yaml_emitter_emit(&self.emitter, &event) == 0:
                error = self._emitter_error()
                raise error
            self.closed = 1

    def serialize(self, node):
        cdef yaml_event_t event
        cdef yaml_version_directive_t version_directive_value
        cdef yaml_version_directive_t *version_directive
        cdef yaml_tag_directive_t tag_directives_value[128]
        cdef yaml_tag_directive_t *tag_directives_start
        cdef yaml_tag_directive_t *tag_directives_end
        if self.closed == -1:
            if PY_MAJOR_VERSION < 3:
                raise SerializerError("serializer is not opened")
            else:
                raise SerializerError(u"serializer is not opened")
        elif self.closed == 1:
            if PY_MAJOR_VERSION < 3:
                raise SerializerError("serializer is closed")
            else:
                raise SerializerError(u"serializer is closed")
        cache = []
        version_directive = NULL
        if self.use_version:
            version_directive_value.major = self.use_version[0]
            version_directive_value.minor = self.use_version[1]
            version_directive = &version_directive_value
        tag_directives_start = NULL
        tag_directives_end = NULL
        if self.use_tags:
            if len(self.use_tags) > 128:
                if PY_MAJOR_VERSION < 3:
                    raise ValueError("too many tags")
                else:
                    raise ValueError(u"too many tags")
            tag_directives_start = tag_directives_value
            tag_directives_end = tag_directives_value
            for handle in self.use_tags:
                prefix = self.use_tags[handle]
                if PyUnicode_CheckExact(handle):
                    handle = PyUnicode_AsUTF8String(handle)
                    cache.append(handle)
                if not PyString_CheckExact(handle):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("tag handle must be a string")
                    else:
                        raise TypeError(u"tag handle must be a string")
                tag_directives_end.handle = PyString_AS_STRING(handle)
                if PyUnicode_CheckExact(prefix):
                    prefix = PyUnicode_AsUTF8String(prefix)
                    cache.append(prefix)
                if not PyString_CheckExact(prefix):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("tag prefix must be a string")
                    else:
                        raise TypeError(u"tag prefix must be a string")
                tag_directives_end.prefix = PyString_AS_STRING(prefix)
                tag_directives_end = tag_directives_end+1
        if yaml_document_start_event_initialize(&event, version_directive,
                tag_directives_start, tag_directives_end,
                self.document_start_implicit) == 0:
            raise MemoryError
        if yaml_emitter_emit(&self.emitter, &event) == 0:
            error = self._emitter_error()
            raise error
        self._anchor_node(node)
        self._serialize_node(node, None, None)
        yaml_document_end_event_initialize(&event, self.document_end_implicit)
        if yaml_emitter_emit(&self.emitter, &event) == 0:
            error = self._emitter_error()
            raise error
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_alias_id = 0

    cdef int _anchor_node(self, object node) except 0:
        if node in self.anchors:
            if self.anchors[node] is None:
                self.last_alias_id = self.last_alias_id+1
                self.anchors[node] = u"id%03d" % self.last_alias_id
        else:
            self.anchors[node] = None
            node_class = node.__class__
            if node_class is SequenceNode:
                for item in node.value:
                    self._anchor_node(item)
            elif node_class is MappingNode:
                for key, value in node.value:
                    self._anchor_node(key)
                    self._anchor_node(value)
        return 1

    cdef int _serialize_node(self, object node, object parent, object index) except 0:
        cdef yaml_event_t event
        cdef int implicit
        cdef int plain_implicit
        cdef int quoted_implicit
        cdef char *anchor
        cdef char *tag
        cdef char *value
        cdef int length
        cdef int item_index
        cdef yaml_scalar_style_t scalar_style
        cdef yaml_sequence_style_t sequence_style
        cdef yaml_mapping_style_t mapping_style
        anchor_object = self.anchors[node]
        anchor = NULL
        if anchor_object is not None:
            if PyUnicode_CheckExact(anchor_object):
                anchor_object = PyUnicode_AsUTF8String(anchor_object)
            if not PyString_CheckExact(anchor_object):
                if PY_MAJOR_VERSION < 3:
                    raise TypeError("anchor must be a string")
                else:
                    raise TypeError(u"anchor must be a string")
            anchor = PyString_AS_STRING(anchor_object)
        if node in self.serialized_nodes:
            if yaml_alias_event_initialize(&event, anchor) == 0:
                raise MemoryError
            if yaml_emitter_emit(&self.emitter, &event) == 0:
                error = self._emitter_error()
                raise error
        else:
            node_class = node.__class__
            self.serialized_nodes[node] = True
            self.descend_resolver(parent, index)
            if node_class is ScalarNode:
                plain_implicit = 0
                quoted_implicit = 0
                tag_object = node.tag
                if self.resolve(ScalarNode, node.value, (True, False)) == tag_object:
                    plain_implicit = 1
                if self.resolve(ScalarNode, node.value, (False, True)) == tag_object:
                    quoted_implicit = 1
                tag = NULL
                if tag_object is not None:
                    if PyUnicode_CheckExact(tag_object):
                        tag_object = PyUnicode_AsUTF8String(tag_object)
                    if not PyString_CheckExact(tag_object):
                        if PY_MAJOR_VERSION < 3:
                            raise TypeError("tag must be a string")
                        else:
                            raise TypeError(u"tag must be a string")
                    tag = PyString_AS_STRING(tag_object)
                value_object = node.value
                if PyUnicode_CheckExact(value_object):
                    value_object = PyUnicode_AsUTF8String(value_object)
                if not PyString_CheckExact(value_object):
                    if PY_MAJOR_VERSION < 3:
                        raise TypeError("value must be a string")
                    else:
                        raise TypeError(u"value must be a string")
                value = PyString_AS_STRING(value_object)
                length = PyString_GET_SIZE(value_object)
                style_object = node.style
                scalar_style = YAML_PLAIN_SCALAR_STYLE
                if style_object == "'" or style_object == u"'":
                    scalar_style = YAML_SINGLE_QUOTED_SCALAR_STYLE
                elif style_object == "\"" or style_object == u"\"":
                    scalar_style = YAML_DOUBLE_QUOTED_SCALAR_STYLE
                elif style_object == "|" or style_object == u"|":
                    scalar_style = YAML_LITERAL_SCALAR_STYLE
                elif style_object == ">" or style_object == u">":
                    scalar_style = YAML_FOLDED_SCALAR_STYLE
                if yaml_scalar_event_initialize(&event, anchor, tag, value, length,
                        plain_implicit, quoted_implicit, scalar_style) == 0:
                    raise MemoryError
                if yaml_emitter_emit(&self.emitter, &event) == 0:
                    error = self._emitter_error()
                    raise error
            elif node_class is SequenceNode:
                implicit = 0
                tag_object = node.tag
                if self.resolve(SequenceNode, node.value, True) == tag_object:
                    implicit = 1
                tag = NULL
                if tag_object is not None:
                    if PyUnicode_CheckExact(tag_object):
                        tag_object = PyUnicode_AsUTF8String(tag_object)
                    if not PyString_CheckExact(tag_object):
                        if PY_MAJOR_VERSION < 3:
                            raise TypeError("tag must be a string")
                        else:
                            raise TypeError(u"tag must be a string")
                    tag = PyString_AS_STRING(tag_object)
                sequence_style = YAML_BLOCK_SEQUENCE_STYLE
                if node.flow_style:
                    sequence_style = YAML_FLOW_SEQUENCE_STYLE
                if yaml_sequence_start_event_initialize(&event, anchor, tag,
                        implicit, sequence_style) == 0:
                    raise MemoryError
                if yaml_emitter_emit(&self.emitter, &event) == 0:
                    error = self._emitter_error()
                    raise error
                item_index = 0
                for item in node.value:
                    self._serialize_node(item, node, item_index)
                    item_index = item_index+1
                yaml_sequence_end_event_initialize(&event)
                if yaml_emitter_emit(&self.emitter, &event) == 0:
                    error = self._emitter_error()
                    raise error
            elif node_class is MappingNode:
                implicit = 0
                tag_object = node.tag
                if self.resolve(MappingNode, node.value, True) == tag_object:
                    implicit = 1
                tag = NULL
                if tag_object is not None:
                    if PyUnicode_CheckExact(tag_object):
                        tag_object = PyUnicode_AsUTF8String(tag_object)
                    if not PyString_CheckExact(tag_object):
                        if PY_MAJOR_VERSION < 3:
                            raise TypeError("tag must be a string")
                        else:
                            raise TypeError(u"tag must be a string")
                    tag = PyString_AS_STRING(tag_object)
                mapping_style = YAML_BLOCK_MAPPING_STYLE
                if node.flow_style:
                    mapping_style = YAML_FLOW_MAPPING_STYLE
                if yaml_mapping_start_event_initialize(&event, anchor, tag,
                        implicit, mapping_style) == 0:
                    raise MemoryError
                if yaml_emitter_emit(&self.emitter, &event) == 0:
                    error = self._emitter_error()
                    raise error
                for item_key, item_value in node.value:
                    self._serialize_node(item_key, node, None)
                    self._serialize_node(item_value, node, item_key)
                yaml_mapping_end_event_initialize(&event)
                if yaml_emitter_emit(&self.emitter, &event) == 0:
                    error = self._emitter_error()
                    raise error
            self.ascend_resolver()
        return 1

cdef int output_handler(void *data, char *buffer, int size) except 0:
    cdef CEmitter emitter
    emitter = <CEmitter>data
    if emitter.dump_unicode == 0:
        value = PyString_FromStringAndSize(buffer, size)
    else:
        value = PyUnicode_DecodeUTF8(buffer, size, 'strict')
    emitter.stream.write(value)
    return 1

