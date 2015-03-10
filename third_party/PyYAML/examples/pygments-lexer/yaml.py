
"""
yaml.py

Lexer for YAML, a human-friendly data serialization language
(http://yaml.org/).

Written by Kirill Simonov <xi@resolvent.net>.

License: Whatever suitable for inclusion into the Pygments package.
"""

from pygments.lexer import  \
        ExtendedRegexLexer, LexerContext, include, bygroups
from pygments.token import  \
        Text, Comment, Punctuation, Name, Literal

__all__ = ['YAMLLexer']


class YAMLLexerContext(LexerContext):
    """Indentation context for the YAML lexer."""

    def __init__(self, *args, **kwds):
        super(YAMLLexerContext, self).__init__(*args, **kwds)
        self.indent_stack = []
        self.indent = -1
        self.next_indent = 0
        self.block_scalar_indent = None


def something(TokenClass):
    """Do not produce empty tokens."""
    def callback(lexer, match, context):
        text = match.group()
        if not text:
            return
        yield match.start(), TokenClass, text
        context.pos = match.end()
    return callback

def reset_indent(TokenClass):
    """Reset the indentation levels."""
    def callback(lexer, match, context):
        text = match.group()
        context.indent_stack = []
        context.indent = -1
        context.next_indent = 0
        context.block_scalar_indent = None
        yield match.start(), TokenClass, text
        context.pos = match.end()
    return callback

def save_indent(TokenClass, start=False):
    """Save a possible indentation level."""
    def callback(lexer, match, context):
        text = match.group()
        extra = ''
        if start:
            context.next_indent = len(text)
            if context.next_indent < context.indent:
                while context.next_indent < context.indent:
                    context.indent = context.indent_stack.pop()
                if context.next_indent > context.indent:
                    extra = text[context.indent:]
                    text = text[:context.indent]
        else:
            context.next_indent += len(text)
        if text:
            yield match.start(), TokenClass, text
        if extra:
            yield match.start()+len(text), TokenClass.Error, extra
        context.pos = match.end()
    return callback

def set_indent(TokenClass, implicit=False):
    """Set the previously saved indentation level."""
    def callback(lexer, match, context):
        text = match.group()
        if context.indent < context.next_indent:
            context.indent_stack.append(context.indent)
            context.indent = context.next_indent
        if not implicit:
            context.next_indent += len(text)
        yield match.start(), TokenClass, text
        context.pos = match.end()
    return callback

def set_block_scalar_indent(TokenClass):
    """Set an explicit indentation level for a block scalar."""
    def callback(lexer, match, context):
        text = match.group()
        context.block_scalar_indent = None
        if not text:
            return
        increment = match.group(1)
        if increment:
            current_indent = max(context.indent, 0)
            increment = int(increment)
            context.block_scalar_indent = current_indent + increment
        if text:
            yield match.start(), TokenClass, text
            context.pos = match.end()
    return callback

def parse_block_scalar_empty_line(IndentTokenClass, ContentTokenClass):
    """Process an empty line in a block scalar."""
    def callback(lexer, match, context):
        text = match.group()
        if (context.block_scalar_indent is None or
                len(text) <= context.block_scalar_indent):
            if text:
                yield match.start(), IndentTokenClass, text
        else:
            indentation = text[:context.block_scalar_indent]
            content = text[context.block_scalar_indent:]
            yield match.start(), IndentTokenClass, indentation
            yield (match.start()+context.block_scalar_indent,
                    ContentTokenClass, content)
        context.pos = match.end()
    return callback

def parse_block_scalar_indent(TokenClass):
    """Process indentation spaces in a block scalar."""
    def callback(lexer, match, context):
        text = match.group()
        if context.block_scalar_indent is None:
            if len(text) <= max(context.indent, 0):
                context.stack.pop()
                context.stack.pop()
                return
            context.block_scalar_indent = len(text)
        else:
            if len(text) < context.block_scalar_indent:
                context.stack.pop()
                context.stack.pop()
                return
        if text:
            yield match.start(), TokenClass, text
            context.pos = match.end()
    return callback

def parse_plain_scalar_indent(TokenClass):
    """Process indentation spaces in a plain scalar."""
    def callback(lexer, match, context):
        text = match.group()
        if len(text) <= context.indent:
            context.stack.pop()
            context.stack.pop()
            return
        if text:
            yield match.start(), TokenClass, text
            context.pos = match.end()
    return callback


class YAMLLexer(ExtendedRegexLexer):
    """Lexer for the YAML language."""

    name = 'YAML'
    aliases = ['yaml']
    filenames = ['*.yaml', '*.yml']
    mimetypes = ['text/x-yaml']

    tokens = {

        # the root rules
        'root': [
            # ignored whitespaces
            (r'[ ]+(?=#|$)', Text.Blank),
            # line breaks
            (r'\n+', Text.Break),
            # a comment
            (r'#[^\n]*', Comment.Single),
            # the '%YAML' directive
            (r'^%YAML(?=[ ]|$)', reset_indent(Name.Directive),
                'yaml-directive'),
            # the %TAG directive
            (r'^%TAG(?=[ ]|$)', reset_indent(Name.Directive),
                'tag-directive'),
            # document start and document end indicators
            (r'^(?:---|\.\.\.)(?=[ ]|$)',
                reset_indent(Punctuation.Document), 'block-line'),
            # indentation spaces
            (r'[ ]*(?![ \t\n\r\f\v]|$)',
                save_indent(Text.Indent, start=True),
                ('block-line', 'indentation')),
        ],

        # trailing whitespaces after directives or a block scalar indicator
        'ignored-line': [
            # ignored whitespaces
            (r'[ ]+(?=#|$)', Text.Blank),
            # a comment
            (r'#[^\n]*', Comment.Single),
            # line break
            (r'\n', Text.Break, '#pop:2'),
        ],

        # the %YAML directive
        'yaml-directive': [
            # the version number
            (r'([ ]+)([0-9]+\.[0-9]+)',
                bygroups(Text.Blank, Literal.Version), 'ignored-line'),
        ],

        # the %YAG directive
        'tag-directive': [
            # a tag handle and the corresponding prefix
            (r'([ ]+)(!|![0-9A-Za-z_-]*!)'
                r'([ ]+)(!|!?[0-9A-Za-z;/?:@&=+$,_.!~*\'()\[\]%-]+)',
                bygroups(Text.Blank, Name.Type, Text.Blank, Name.Type),
                'ignored-line'),
        ],

        # block scalar indicators and indentation spaces
        'indentation': [
            # trailing whitespaces are ignored
            (r'[ ]*$', something(Text.Blank), '#pop:2'),
            # whitespaces preceeding block collection indicators
            (r'[ ]+(?=[?:-](?:[ ]|$))', save_indent(Text.Indent)),
            # block collection indicators
            (r'[?:-](?=[ ]|$)', set_indent(Punctuation.Indicator)),
            # the beginning a block line
            (r'[ ]*', save_indent(Text.Indent), '#pop'),
        ],

        # an indented line in the block context
        'block-line': [
            # the line end
            (r'[ ]*(?=#|$)', something(Text.Blank), '#pop'),
            # whitespaces separating tokens
            (r'[ ]+', Text.Blank),
            # tags, anchors and aliases,
            include('descriptors'),
            # block collections and scalars
            include('block-nodes'),
            # flow collections and quoted scalars
            include('flow-nodes'),
            # a plain scalar
            (r'(?=[^ \t\n\r\f\v?:,\[\]{}#&*!|>\'"%@`-]|[?:-][^ \t\n\r\f\v])',
                something(Literal.Scalar.Plain),
                'plain-scalar-in-block-context'),
        ],

        # tags, anchors, aliases
        'descriptors' : [
            # a full-form tag
            (r'!<[0-9A-Za-z;/?:@&=+$,_.!~*\'()\[\]%-]+>', Name.Type),
            # a tag in the form '!', '!suffix' or '!handle!suffix'
            (r'!(?:[0-9A-Za-z_-]+)?'
                r'(?:![0-9A-Za-z;/?:@&=+$,_.!~*\'()\[\]%-]+)?', Name.Type),
            # an anchor
            (r'&[0-9A-Za-z_-]+', Name.Anchor),
            # an alias
            (r'\*[0-9A-Za-z_-]+', Name.Alias),
        ],

        # block collections and scalars
        'block-nodes': [
            # implicit key
            (r':(?=[ ]|$)', set_indent(Punctuation.Indicator, implicit=True)),
            # literal and folded scalars
            (r'[|>]', Punctuation.Indicator,
                ('block-scalar-content', 'block-scalar-header')),
        ],

        # flow collections and quoted scalars
        'flow-nodes': [
            # a flow sequence
            (r'\[', Punctuation.Indicator, 'flow-sequence'),
            # a flow mapping
            (r'\{', Punctuation.Indicator, 'flow-mapping'),
            # a single-quoted scalar
            (r'\'', Literal.Scalar.Flow.Quote, 'single-quoted-scalar'),
            # a double-quoted scalar
            (r'\"', Literal.Scalar.Flow.Quote, 'double-quoted-scalar'),
        ],

        # the content of a flow collection
        'flow-collection': [
            # whitespaces
            (r'[ ]+', Text.Blank),
            # line breaks
            (r'\n+', Text.Break),
            # a comment
            (r'#[^\n]*', Comment.Single),
            # simple indicators
            (r'[?:,]', Punctuation.Indicator),
            # tags, anchors and aliases
            include('descriptors'),
            # nested collections and quoted scalars
            include('flow-nodes'),
            # a plain scalar
            (r'(?=[^ \t\n\r\f\v?:,\[\]{}#&*!|>\'"%@`])',
                something(Literal.Scalar.Plain),
                'plain-scalar-in-flow-context'),
        ],

        # a flow sequence indicated by '[' and ']'
        'flow-sequence': [
            # include flow collection rules
            include('flow-collection'),
            # the closing indicator
            (r'\]', Punctuation.Indicator, '#pop'),
        ],

        # a flow mapping indicated by '{' and '}'
        'flow-mapping': [
            # include flow collection rules
            include('flow-collection'),
            # the closing indicator
            (r'\}', Punctuation.Indicator, '#pop'),
        ],

        # block scalar lines
        'block-scalar-content': [
            # line break
            (r'\n', Text.Break),
            # empty line
            (r'^[ ]+$',
                parse_block_scalar_empty_line(Text.Indent,
                    Literal.Scalar.Block)),
            # indentation spaces (we may leave the state here)
            (r'^[ ]*', parse_block_scalar_indent(Text.Indent)),
            # line content
            (r'[^\n\r\f\v]+', Literal.Scalar.Block),
        ],

        # the content of a literal or folded scalar
        'block-scalar-header': [
            # indentation indicator followed by chomping flag
            (r'([1-9])?[+-]?(?=[ ]|$)',
                set_block_scalar_indent(Punctuation.Indicator),
                'ignored-line'),
            # chomping flag followed by indentation indicator
            (r'[+-]?([1-9])?(?=[ ]|$)',
                set_block_scalar_indent(Punctuation.Indicator),
                'ignored-line'),
        ],

        # ignored and regular whitespaces in quoted scalars
        'quoted-scalar-whitespaces': [
            # leading and trailing whitespaces are ignored
            (r'^[ ]+|[ ]+$', Text.Blank),
            # line breaks are ignored
            (r'\n+', Text.Break),
            # other whitespaces are a part of the value
            (r'[ ]+', Literal.Scalar.Flow),
        ],

        # single-quoted scalars
        'single-quoted-scalar': [
            # include whitespace and line break rules
            include('quoted-scalar-whitespaces'),
            # escaping of the quote character
            (r'\'\'', Literal.Scalar.Flow.Escape),
            # regular non-whitespace characters
            (r'[^ \t\n\r\f\v\']+', Literal.Scalar.Flow),
            # the closing quote
            (r'\'', Literal.Scalar.Flow.Quote, '#pop'),
        ],

        # double-quoted scalars
        'double-quoted-scalar': [
            # include whitespace and line break rules
            include('quoted-scalar-whitespaces'),
            # escaping of special characters
            (r'\\[0abt\tn\nvfre "\\N_LP]', Literal.Scalar.Flow.Escape),
            # escape codes
            (r'\\(?:x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})',
                Literal.Scalar.Flow.Escape),
            # regular non-whitespace characters
            (r'[^ \t\n\r\f\v\"\\]+', Literal.Scalar.Flow),
            # the closing quote
            (r'"', Literal.Scalar.Flow.Quote, '#pop'),
        ],

        # the beginning of a new line while scanning a plain scalar
        'plain-scalar-in-block-context-new-line': [
            # empty lines
            (r'^[ ]+$', Text.Blank),
            # line breaks
            (r'\n+', Text.Break),
            # document start and document end indicators
            (r'^(?=---|\.\.\.)', something(Punctuation.Document), '#pop:3'),
            # indentation spaces (we may leave the block line state here)
            (r'^[ ]*', parse_plain_scalar_indent(Text.Indent), '#pop'),
        ],

        # a plain scalar in the block context
        'plain-scalar-in-block-context': [
            # the scalar ends with the ':' indicator
            (r'[ ]*(?=:[ ]|:$)', something(Text.Blank), '#pop'),
            # the scalar ends with whitespaces followed by a comment
            (r'[ ]+(?=#)', Text.Blank, '#pop'),
            # trailing whitespaces are ignored
            (r'[ ]+$', Text.Blank),
            # line breaks are ignored
            (r'\n+', Text.Break, 'plain-scalar-in-block-context-new-line'),
            # other whitespaces are a part of the value
            (r'[ ]+', Literal.Scalar.Plain),
            # regular non-whitespace characters
            (r'(?::(?![ \t\n\r\f\v])|[^ \t\n\r\f\v:])+',
                Literal.Scalar.Plain),
        ],

        # a plain scalar is the flow context
        'plain-scalar-in-flow-context': [
            # the scalar ends with an indicator character
            (r'[ ]*(?=[,:?\[\]{}])', something(Text.Blank), '#pop'),
            # the scalar ends with a comment
            (r'[ ]+(?=#)', Text.Blank, '#pop'),
            # leading and trailing whitespaces are ignored
            (r'^[ ]+|[ ]+$', Text.Blank),
            # line breaks are ignored
            (r'\n+', Text.Break),
            # other whitespaces are a part of the value
            (r'[ ]+', Literal.Scalar.Plain),
            # regular non-whitespace characters
            (r'[^ \t\n\r\f\v,:?\[\]{}]+', Literal.Scalar.Plain),
        ],

    }

    def get_tokens_unprocessed(self, text=None, context=None):
        if context is None:
            context = YAMLLexerContext(text, 0)
        return super(YAMLLexer, self).get_tokens_unprocessed(text, context)


