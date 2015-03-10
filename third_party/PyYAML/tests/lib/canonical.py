
import yaml, yaml.composer, yaml.constructor, yaml.resolver

class CanonicalError(yaml.YAMLError):
    pass

class CanonicalScanner:

    def __init__(self, data):
        try:
            self.data = unicode(data, 'utf-8')+u'\0'
        except UnicodeDecodeError:
            raise CanonicalError("utf-8 stream is expected")
        self.index = 0
        self.tokens = []
        self.scanned = False

    def check_token(self, *choices):
        if not self.scanned:
            self.scan()
        if self.tokens:
            if not choices:
                return True
            for choice in choices:
                if isinstance(self.tokens[0], choice):
                    return True
        return False

    def peek_token(self):
        if not self.scanned:
            self.scan()
        if self.tokens:
            return self.tokens[0]

    def get_token(self, choice=None):
        if not self.scanned:
            self.scan()
        token = self.tokens.pop(0)
        if choice and not isinstance(token, choice):
            raise CanonicalError("unexpected token "+repr(token))
        return token

    def get_token_value(self):
        token = self.get_token()
        return token.value

    def scan(self):
        self.tokens.append(yaml.StreamStartToken(None, None))
        while True:
            self.find_token()
            ch = self.data[self.index]
            if ch == u'\0':
                self.tokens.append(yaml.StreamEndToken(None, None))
                break
            elif ch == u'%':
                self.tokens.append(self.scan_directive())
            elif ch == u'-' and self.data[self.index:self.index+3] == u'---':
                self.index += 3
                self.tokens.append(yaml.DocumentStartToken(None, None))
            elif ch == u'[':
                self.index += 1
                self.tokens.append(yaml.FlowSequenceStartToken(None, None))
            elif ch == u'{':
                self.index += 1
                self.tokens.append(yaml.FlowMappingStartToken(None, None))
            elif ch == u']':
                self.index += 1
                self.tokens.append(yaml.FlowSequenceEndToken(None, None))
            elif ch == u'}':
                self.index += 1
                self.tokens.append(yaml.FlowMappingEndToken(None, None))
            elif ch == u'?':
                self.index += 1
                self.tokens.append(yaml.KeyToken(None, None))
            elif ch == u':':
                self.index += 1
                self.tokens.append(yaml.ValueToken(None, None))
            elif ch == u',':
                self.index += 1
                self.tokens.append(yaml.FlowEntryToken(None, None))
            elif ch == u'*' or ch == u'&':
                self.tokens.append(self.scan_alias())
            elif ch == u'!':
                self.tokens.append(self.scan_tag())
            elif ch == u'"':
                self.tokens.append(self.scan_scalar())
            else:
                raise CanonicalError("invalid token")
        self.scanned = True

    DIRECTIVE = u'%YAML 1.1'

    def scan_directive(self):
        if self.data[self.index:self.index+len(self.DIRECTIVE)] == self.DIRECTIVE and \
                self.data[self.index+len(self.DIRECTIVE)] in u' \n\0':
            self.index += len(self.DIRECTIVE)
            return yaml.DirectiveToken('YAML', (1, 1), None, None)
        else:
            raise CanonicalError("invalid directive")

    def scan_alias(self):
        if self.data[self.index] == u'*':
            TokenClass = yaml.AliasToken
        else:
            TokenClass = yaml.AnchorToken
        self.index += 1
        start = self.index
        while self.data[self.index] not in u', \n\0':
            self.index += 1
        value = self.data[start:self.index]
        return TokenClass(value, None, None)

    def scan_tag(self):
        self.index += 1
        start = self.index
        while self.data[self.index] not in u' \n\0':
            self.index += 1
        value = self.data[start:self.index]
        if not value:
            value = u'!'
        elif value[0] == u'!':
            value = 'tag:yaml.org,2002:'+value[1:]
        elif value[0] == u'<' and value[-1] == u'>':
            value = value[1:-1]
        else:
            value = u'!'+value
        return yaml.TagToken(value, None, None)

    QUOTE_CODES = {
        'x': 2,
        'u': 4,
        'U': 8,
    }

    QUOTE_REPLACES = {
        u'\\': u'\\',
        u'\"': u'\"',
        u' ': u' ',
        u'a': u'\x07',
        u'b': u'\x08',
        u'e': u'\x1B',
        u'f': u'\x0C',
        u'n': u'\x0A',
        u'r': u'\x0D',
        u't': u'\x09',
        u'v': u'\x0B',
        u'N': u'\u0085',
        u'L': u'\u2028',
        u'P': u'\u2029',
        u'_': u'_',
        u'0': u'\x00',

    }

    def scan_scalar(self):
        self.index += 1
        chunks = []
        start = self.index
        ignore_spaces = False
        while self.data[self.index] != u'"':
            if self.data[self.index] == u'\\':
                ignore_spaces = False
                chunks.append(self.data[start:self.index])
                self.index += 1
                ch = self.data[self.index]
                self.index += 1
                if ch == u'\n':
                    ignore_spaces = True
                elif ch in self.QUOTE_CODES:
                    length = self.QUOTE_CODES[ch]
                    code = int(self.data[self.index:self.index+length], 16)
                    chunks.append(unichr(code))
                    self.index += length
                else:
                    if ch not in self.QUOTE_REPLACES:
                        raise CanonicalError("invalid escape code")
                    chunks.append(self.QUOTE_REPLACES[ch])
                start = self.index
            elif self.data[self.index] == u'\n':
                chunks.append(self.data[start:self.index])
                chunks.append(u' ')
                self.index += 1
                start = self.index
                ignore_spaces = True
            elif ignore_spaces and self.data[self.index] == u' ':
                self.index += 1
                start = self.index
            else:
                ignore_spaces = False
                self.index += 1
        chunks.append(self.data[start:self.index])
        self.index += 1
        return yaml.ScalarToken(u''.join(chunks), False, None, None)

    def find_token(self):
        found = False
        while not found:
            while self.data[self.index] in u' \t':
                self.index += 1
            if self.data[self.index] == u'#':
                while self.data[self.index] != u'\n':
                    self.index += 1
            if self.data[self.index] == u'\n':
                self.index += 1
            else:
                found = True

class CanonicalParser:

    def __init__(self):
        self.events = []
        self.parsed = False

    def dispose(self):
        pass

    # stream: STREAM-START document* STREAM-END
    def parse_stream(self):
        self.get_token(yaml.StreamStartToken)
        self.events.append(yaml.StreamStartEvent(None, None))
        while not self.check_token(yaml.StreamEndToken):
            if self.check_token(yaml.DirectiveToken, yaml.DocumentStartToken):
                self.parse_document()
            else:
                raise CanonicalError("document is expected, got "+repr(self.tokens[0]))
        self.get_token(yaml.StreamEndToken)
        self.events.append(yaml.StreamEndEvent(None, None))

    # document: DIRECTIVE? DOCUMENT-START node
    def parse_document(self):
        node = None
        if self.check_token(yaml.DirectiveToken):
            self.get_token(yaml.DirectiveToken)
        self.get_token(yaml.DocumentStartToken)
        self.events.append(yaml.DocumentStartEvent(None, None))
        self.parse_node()
        self.events.append(yaml.DocumentEndEvent(None, None))

    # node: ALIAS | ANCHOR? TAG? (SCALAR|sequence|mapping)
    def parse_node(self):
        if self.check_token(yaml.AliasToken):
            self.events.append(yaml.AliasEvent(self.get_token_value(), None, None))
        else:
            anchor = None
            if self.check_token(yaml.AnchorToken):
                anchor = self.get_token_value()
            tag = None
            if self.check_token(yaml.TagToken):
                tag = self.get_token_value()
            if self.check_token(yaml.ScalarToken):
                self.events.append(yaml.ScalarEvent(anchor, tag, (False, False), self.get_token_value(), None, None))
            elif self.check_token(yaml.FlowSequenceStartToken):
                self.events.append(yaml.SequenceStartEvent(anchor, tag, None, None))
                self.parse_sequence()
            elif self.check_token(yaml.FlowMappingStartToken):
                self.events.append(yaml.MappingStartEvent(anchor, tag, None, None))
                self.parse_mapping()
            else:
                raise CanonicalError("SCALAR, '[', or '{' is expected, got "+repr(self.tokens[0]))

    # sequence: SEQUENCE-START (node (ENTRY node)*)? ENTRY? SEQUENCE-END
    def parse_sequence(self):
        self.get_token(yaml.FlowSequenceStartToken)
        if not self.check_token(yaml.FlowSequenceEndToken):
            self.parse_node()
            while not self.check_token(yaml.FlowSequenceEndToken):
                self.get_token(yaml.FlowEntryToken)
                if not self.check_token(yaml.FlowSequenceEndToken):
                    self.parse_node()
        self.get_token(yaml.FlowSequenceEndToken)
        self.events.append(yaml.SequenceEndEvent(None, None))

    # mapping: MAPPING-START (map_entry (ENTRY map_entry)*)? ENTRY? MAPPING-END
    def parse_mapping(self):
        self.get_token(yaml.FlowMappingStartToken)
        if not self.check_token(yaml.FlowMappingEndToken):
            self.parse_map_entry()
            while not self.check_token(yaml.FlowMappingEndToken):
                self.get_token(yaml.FlowEntryToken)
                if not self.check_token(yaml.FlowMappingEndToken):
                    self.parse_map_entry()
        self.get_token(yaml.FlowMappingEndToken)
        self.events.append(yaml.MappingEndEvent(None, None))

    # map_entry: KEY node VALUE node
    def parse_map_entry(self):
        self.get_token(yaml.KeyToken)
        self.parse_node()
        self.get_token(yaml.ValueToken)
        self.parse_node()

    def parse(self):
        self.parse_stream()
        self.parsed = True

    def get_event(self):
        if not self.parsed:
            self.parse()
        return self.events.pop(0)

    def check_event(self, *choices):
        if not self.parsed:
            self.parse()
        if self.events:
            if not choices:
                return True
            for choice in choices:
                if isinstance(self.events[0], choice):
                    return True
        return False

    def peek_event(self):
        if not self.parsed:
            self.parse()
        return self.events[0]

class CanonicalLoader(CanonicalScanner, CanonicalParser,
        yaml.composer.Composer, yaml.constructor.Constructor, yaml.resolver.Resolver):

    def __init__(self, stream):
        if hasattr(stream, 'read'):
            stream = stream.read()
        CanonicalScanner.__init__(self, stream)
        CanonicalParser.__init__(self)
        yaml.composer.Composer.__init__(self)
        yaml.constructor.Constructor.__init__(self)
        yaml.resolver.Resolver.__init__(self)

yaml.CanonicalLoader = CanonicalLoader

def canonical_scan(stream):
    return yaml.scan(stream, Loader=CanonicalLoader)

yaml.canonical_scan = canonical_scan

def canonical_parse(stream):
    return yaml.parse(stream, Loader=CanonicalLoader)

yaml.canonical_parse = canonical_parse

def canonical_compose(stream):
    return yaml.compose(stream, Loader=CanonicalLoader)

yaml.canonical_compose = canonical_compose

def canonical_compose_all(stream):
    return yaml.compose_all(stream, Loader=CanonicalLoader)

yaml.canonical_compose_all = canonical_compose_all

def canonical_load(stream):
    return yaml.load(stream, Loader=CanonicalLoader)

yaml.canonical_load = canonical_load

def canonical_load_all(stream):
    return yaml.load_all(stream, Loader=CanonicalLoader)

yaml.canonical_load_all = canonical_load_all

