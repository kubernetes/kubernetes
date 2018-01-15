package yaml

import (
	"bytes"
	"fmt"
)

// Introduction
// ************
//
// The following notes assume that you are familiar with the YAML specification
// (http://yaml.org/spec/1.2/spec.html).  We mostly follow it, although in
// some cases we are less restrictive that it requires.
//
// The process of transforming a YAML stream into a sequence of events is
// divided on two steps: Scanning and Parsing.
//
// The Scanner transforms the input stream into a sequence of tokens, while the
// parser transform the sequence of tokens produced by the Scanner into a
// sequence of parsing events.
//
// The Scanner is rather clever and complicated. The Parser, on the contrary,
// is a straightforward implementation of a recursive-descendant parser (or,
// LL(1) parser, as it is usually called).
//
// Actually there are two issues of Scanning that might be called "clever", the
// rest is quite straightforward.  The issues are "block collection start" and
// "simple keys".  Both issues are explained below in details.
//
// Here the Scanning step is explained and implemented.  We start with the list
// of all the tokens produced by the Scanner together with short descriptions.
//
// Now, tokens:
//
//      STREAM-START(encoding)          # The stream start.
//      STREAM-END                      # The stream end.
//      VERSION-DIRECTIVE(major,minor)  # The '%YAML' directive.
//      TAG-DIRECTIVE(handle,prefix)    # The '%TAG' directive.
//      DOCUMENT-START                  # '---'
//      DOCUMENT-END                    # '...'
//      BLOCK-SEQUENCE-START            # Indentation increase denoting a block
//      BLOCK-MAPPING-START             # sequence or a block mapping.
//      BLOCK-END                       # Indentation decrease.
//      FLOW-SEQUENCE-START             # '['
//      FLOW-SEQUENCE-END               # ']'
//      BLOCK-SEQUENCE-START            # '{'
//      BLOCK-SEQUENCE-END              # '}'
//      BLOCK-ENTRY                     # '-'
//      FLOW-ENTRY                      # ','
//      KEY                             # '?' or nothing (simple keys).
//      VALUE                           # ':'
//      ALIAS(anchor)                   # '*anchor'
//      ANCHOR(anchor)                  # '&anchor'
//      TAG(handle,suffix)              # '!handle!suffix'
//      SCALAR(value,style)             # A scalar.
//
// The following two tokens are "virtual" tokens denoting the beginning and the
// end of the stream:
//
//      STREAM-START(encoding)
//      STREAM-END
//
// We pass the information about the input stream encoding with the
// STREAM-START token.
//
// The next two tokens are responsible for tags:
//
//      VERSION-DIRECTIVE(major,minor)
//      TAG-DIRECTIVE(handle,prefix)
//
// Example:
//
//      %YAML   1.1
//      %TAG    !   !foo
//      %TAG    !yaml!  tag:yaml.org,2002:
//      ---
//
// The correspoding sequence of tokens:
//
//      STREAM-START(utf-8)
//      VERSION-DIRECTIVE(1,1)
//      TAG-DIRECTIVE("!","!foo")
//      TAG-DIRECTIVE("!yaml","tag:yaml.org,2002:")
//      DOCUMENT-START
//      STREAM-END
//
// Note that the VERSION-DIRECTIVE and TAG-DIRECTIVE tokens occupy a whole
// line.
//
// The document start and end indicators are represented by:
//
//      DOCUMENT-START
//      DOCUMENT-END
//
// Note that if a YAML stream contains an implicit document (without '---'
// and '...' indicators), no DOCUMENT-START and DOCUMENT-END tokens will be
// produced.
//
// In the following examples, we present whole documents together with the
// produced tokens.
//
//      1. An implicit document:
//
//          'a scalar'
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          SCALAR("a scalar",single-quoted)
//          STREAM-END
//
//      2. An explicit document:
//
//          ---
//          'a scalar'
//          ...
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          DOCUMENT-START
//          SCALAR("a scalar",single-quoted)
//          DOCUMENT-END
//          STREAM-END
//
//      3. Several documents in a stream:
//
//          'a scalar'
//          ---
//          'another scalar'
//          ---
//          'yet another scalar'
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          SCALAR("a scalar",single-quoted)
//          DOCUMENT-START
//          SCALAR("another scalar",single-quoted)
//          DOCUMENT-START
//          SCALAR("yet another scalar",single-quoted)
//          STREAM-END
//
// We have already introduced the SCALAR token above.  The following tokens are
// used to describe aliases, anchors, tag, and scalars:
//
//      ALIAS(anchor)
//      ANCHOR(anchor)
//      TAG(handle,suffix)
//      SCALAR(value,style)
//
// The following series of examples illustrate the usage of these tokens:
//
//      1. A recursive sequence:
//
//          &A [ *A ]
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          ANCHOR("A")
//          FLOW-SEQUENCE-START
//          ALIAS("A")
//          FLOW-SEQUENCE-END
//          STREAM-END
//
//      2. A tagged scalar:
//
//          !!float "3.14"  # A good approximation.
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          TAG("!!","float")
//          SCALAR("3.14",double-quoted)
//          STREAM-END
//
//      3. Various scalar styles:
//
//          --- # Implicit empty plain scalars do not produce tokens.
//          --- a plain scalar
//          --- 'a single-quoted scalar'
//          --- "a double-quoted scalar"
//          --- |-
//            a literal scalar
//          --- >-
//            a folded
//            scalar
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          DOCUMENT-START
//          DOCUMENT-START
//          SCALAR("a plain scalar",plain)
//          DOCUMENT-START
//          SCALAR("a single-quoted scalar",single-quoted)
//          DOCUMENT-START
//          SCALAR("a double-quoted scalar",double-quoted)
//          DOCUMENT-START
//          SCALAR("a literal scalar",literal)
//          DOCUMENT-START
//          SCALAR("a folded scalar",folded)
//          STREAM-END
//
// Now it's time to review collection-related tokens. We will start with
// flow collections:
//
//      FLOW-SEQUENCE-START
//      FLOW-SEQUENCE-END
//      FLOW-MAPPING-START
//      FLOW-MAPPING-END
//      FLOW-ENTRY
//      KEY
//      VALUE
//
// The tokens FLOW-SEQUENCE-START, FLOW-SEQUENCE-END, FLOW-MAPPING-START, and
// FLOW-MAPPING-END represent the indicators '[', ']', '{', and '}'
// correspondingly.  FLOW-ENTRY represent the ',' indicator.  Finally the
// indicators '?' and ':', which are used for denoting mapping keys and values,
// are represented by the KEY and VALUE tokens.
//
// The following examples show flow collections:
//
//      1. A flow sequence:
//
//          [item 1, item 2, item 3]
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          FLOW-SEQUENCE-START
//          SCALAR("item 1",plain)
//          FLOW-ENTRY
//          SCALAR("item 2",plain)
//          FLOW-ENTRY
//          SCALAR("item 3",plain)
//          FLOW-SEQUENCE-END
//          STREAM-END
//
//      2. A flow mapping:
//
//          {
//              a simple key: a value,  # Note that the KEY token is produced.
//              ? a complex key: another value,
//          }
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          FLOW-MAPPING-START
//          KEY
//          SCALAR("a simple key",plain)
//          VALUE
//          SCALAR("a value",plain)
//          FLOW-ENTRY
//          KEY
//          SCALAR("a complex key",plain)
//          VALUE
//          SCALAR("another value",plain)
//          FLOW-ENTRY
//          FLOW-MAPPING-END
//          STREAM-END
//
// A simple key is a key which is not denoted by the '?' indicator.  Note that
// the Scanner still produce the KEY token whenever it encounters a simple key.
//
// For scanning block collections, the following tokens are used (note that we
// repeat KEY and VALUE here):
//
//      BLOCK-SEQUENCE-START
//      BLOCK-MAPPING-START
//      BLOCK-END
//      BLOCK-ENTRY
//      KEY
//      VALUE
//
// The tokens BLOCK-SEQUENCE-START and BLOCK-MAPPING-START denote indentation
// increase that precedes a block collection (cf. the INDENT token in Python).
// The token BLOCK-END denote indentation decrease that ends a block collection
// (cf. the DEDENT token in Python).  However YAML has some syntax pecularities
// that makes detections of these tokens more complex.
//
// The tokens BLOCK-ENTRY, KEY, and VALUE are used to represent the indicators
// '-', '?', and ':' correspondingly.
//
// The following examples show how the tokens BLOCK-SEQUENCE-START,
// BLOCK-MAPPING-START, and BLOCK-END are emitted by the Scanner:
//
//      1. Block sequences:
//
//          - item 1
//          - item 2
//          -
//            - item 3.1
//            - item 3.2
//          -
//            key 1: value 1
//            key 2: value 2
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          SCALAR("item 1",plain)
//          BLOCK-ENTRY
//          SCALAR("item 2",plain)
//          BLOCK-ENTRY
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          SCALAR("item 3.1",plain)
//          BLOCK-ENTRY
//          SCALAR("item 3.2",plain)
//          BLOCK-END
//          BLOCK-ENTRY
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("key 1",plain)
//          VALUE
//          SCALAR("value 1",plain)
//          KEY
//          SCALAR("key 2",plain)
//          VALUE
//          SCALAR("value 2",plain)
//          BLOCK-END
//          BLOCK-END
//          STREAM-END
//
//      2. Block mappings:
//
//          a simple key: a value   # The KEY token is produced here.
//          ? a complex key
//          : another value
//          a mapping:
//            key 1: value 1
//            key 2: value 2
//          a sequence:
//            - item 1
//            - item 2
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("a simple key",plain)
//          VALUE
//          SCALAR("a value",plain)
//          KEY
//          SCALAR("a complex key",plain)
//          VALUE
//          SCALAR("another value",plain)
//          KEY
//          SCALAR("a mapping",plain)
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("key 1",plain)
//          VALUE
//          SCALAR("value 1",plain)
//          KEY
//          SCALAR("key 2",plain)
//          VALUE
//          SCALAR("value 2",plain)
//          BLOCK-END
//          KEY
//          SCALAR("a sequence",plain)
//          VALUE
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          SCALAR("item 1",plain)
//          BLOCK-ENTRY
//          SCALAR("item 2",plain)
//          BLOCK-END
//          BLOCK-END
//          STREAM-END
//
// YAML does not always require to start a new block collection from a new
// line.  If the current line contains only '-', '?', and ':' indicators, a new
// block collection may start at the current line.  The following examples
// illustrate this case:
//
//      1. Collections in a sequence:
//
//          - - item 1
//            - item 2
//          - key 1: value 1
//            key 2: value 2
//          - ? complex key
//            : complex value
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          SCALAR("item 1",plain)
//          BLOCK-ENTRY
//          SCALAR("item 2",plain)
//          BLOCK-END
//          BLOCK-ENTRY
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("key 1",plain)
//          VALUE
//          SCALAR("value 1",plain)
//          KEY
//          SCALAR("key 2",plain)
//          VALUE
//          SCALAR("value 2",plain)
//          BLOCK-END
//          BLOCK-ENTRY
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("complex key")
//          VALUE
//          SCALAR("complex value")
//          BLOCK-END
//          BLOCK-END
//          STREAM-END
//
//      2. Collections in a mapping:
//
//          ? a sequence
//          : - item 1
//            - item 2
//          ? a mapping
//          : key 1: value 1
//            key 2: value 2
//
//      Tokens:
//
//          STREAM-START(utf-8)
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("a sequence",plain)
//          VALUE
//          BLOCK-SEQUENCE-START
//          BLOCK-ENTRY
//          SCALAR("item 1",plain)
//          BLOCK-ENTRY
//          SCALAR("item 2",plain)
//          BLOCK-END
//          KEY
//          SCALAR("a mapping",plain)
//          VALUE
//          BLOCK-MAPPING-START
//          KEY
//          SCALAR("key 1",plain)
//          VALUE
//          SCALAR("value 1",plain)
//          KEY
//          SCALAR("key 2",plain)
//          VALUE
//          SCALAR("value 2",plain)
//          BLOCK-END
//          BLOCK-END
//          STREAM-END
//
// YAML also permits non-indented sequences if they are included into a block
// mapping.  In this case, the token BLOCK-SEQUENCE-START is not produced:
//
//      key:
//      - item 1    # BLOCK-SEQUENCE-START is NOT produced here.
//      - item 2
//
// Tokens:
//
//      STREAM-START(utf-8)
//      BLOCK-MAPPING-START
//      KEY
//      SCALAR("key",plain)
//      VALUE
//      BLOCK-ENTRY
//      SCALAR("item 1",plain)
//      BLOCK-ENTRY
//      SCALAR("item 2",plain)
//      BLOCK-END
//

// Ensure that the buffer contains the required number of characters.
// Return true on success, false on failure (reader error or memory error).
func cache(parser *yaml_parser_t, length int) bool {
	// [Go] This was inlined: !cache(A, B) -> unread < B && !update(A, B)
	return parser.unread >= length || yaml_parser_update_buffer(parser, length)
}

// Advance the buffer pointer.
func skip(parser *yaml_parser_t) {
	parser.mark.index++
	parser.mark.column++
	parser.unread--
	parser.buffer_pos += width(parser.buffer[parser.buffer_pos])
}

func skip_line(parser *yaml_parser_t) {
	if is_crlf(parser.buffer, parser.buffer_pos) {
		parser.mark.index += 2
		parser.mark.column = 0
		parser.mark.line++
		parser.unread -= 2
		parser.buffer_pos += 2
	} else if is_break(parser.buffer, parser.buffer_pos) {
		parser.mark.index++
		parser.mark.column = 0
		parser.mark.line++
		parser.unread--
		parser.buffer_pos += width(parser.buffer[parser.buffer_pos])
	}
}

// Copy a character to a string buffer and advance pointers.
func read(parser *yaml_parser_t, s []byte) []byte {
	w := width(parser.buffer[parser.buffer_pos])
	if w == 0 {
		panic("invalid character sequence")
	}
	if len(s) == 0 {
		s = make([]byte, 0, 32)
	}
	if w == 1 && len(s)+w <= cap(s) {
		s = s[:len(s)+1]
		s[len(s)-1] = parser.buffer[parser.buffer_pos]
		parser.buffer_pos++
	} else {
		s = append(s, parser.buffer[parser.buffer_pos:parser.buffer_pos+w]...)
		parser.buffer_pos += w
	}
	parser.mark.index++
	parser.mark.column++
	parser.unread--
	return s
}

// Copy a line break character to a string buffer and advance pointers.
func read_line(parser *yaml_parser_t, s []byte) []byte {
	buf := parser.buffer
	pos := parser.buffer_pos
	switch {
	case buf[pos] == '\r' && buf[pos+1] == '\n':
		// CR LF . LF
		s = append(s, '\n')
		parser.buffer_pos += 2
		parser.mark.index++
		parser.unread--
	case buf[pos] == '\r' || buf[pos] == '\n':
		// CR|LF . LF
		s = append(s, '\n')
		parser.buffer_pos += 1
	case buf[pos] == '\xC2' && buf[pos+1] == '\x85':
		// NEL . LF
		s = append(s, '\n')
		parser.buffer_pos += 2
	case buf[pos] == '\xE2' && buf[pos+1] == '\x80' && (buf[pos+2] == '\xA8' || buf[pos+2] == '\xA9'):
		// LS|PS . LS|PS
		s = append(s, buf[parser.buffer_pos:pos+3]...)
		parser.buffer_pos += 3
	default:
		return s
	}
	parser.mark.index++
	parser.mark.column = 0
	parser.mark.line++
	parser.unread--
	return s
}

// Get the next token.
func yaml_parser_scan(parser *yaml_parser_t, token *yaml_token_t) bool {
	// Erase the token object.
	*token = yaml_token_t{} // [Go] Is this necessary?

	// No tokens after STREAM-END or error.
	if parser.stream_end_produced || parser.error != yaml_NO_ERROR {
		return true
	}

	// Ensure that the tokens queue contains enough tokens.
	if !parser.token_available {
		if !yaml_parser_fetch_more_tokens(parser) {
			return false
		}
	}

	// Fetch the next token from the queue.
	*token = parser.tokens[parser.tokens_head]
	parser.tokens_head++
	parser.tokens_parsed++
	parser.token_available = false

	if token.typ == yaml_STREAM_END_TOKEN {
		parser.stream_end_produced = true
	}
	return true
}

// Set the scanner error and return false.
func yaml_parser_set_scanner_error(parser *yaml_parser_t, context string, context_mark yaml_mark_t, problem string) bool {
	parser.error = yaml_SCANNER_ERROR
	parser.context = context
	parser.context_mark = context_mark
	parser.problem = problem
	parser.problem_mark = parser.mark
	return false
}

func yaml_parser_set_scanner_tag_error(parser *yaml_parser_t, directive bool, context_mark yaml_mark_t, problem string) bool {
	context := "while parsing a tag"
	if directive {
		context = "while parsing a %TAG directive"
	}
	return yaml_parser_set_scanner_error(parser, context, context_mark, problem)
}

func trace(args ...interface{}) func() {
	pargs := append([]interface{}{"+++"}, args...)
	fmt.Println(pargs...)
	pargs = append([]interface{}{"---"}, args...)
	return func() { fmt.Println(pargs...) }
}

// Ensure that the tokens queue contains at least one token which can be
// returned to the Parser.
func yaml_parser_fetch_more_tokens(parser *yaml_parser_t) bool {
	// While we need more tokens to fetch, do it.
	for {
		// Check if we really need to fetch more tokens.
		need_more_tokens := false

		if parser.tokens_head == len(parser.tokens) {
			// Queue is empty.
			need_more_tokens = true
		} else {
			// Check if any potential simple key may occupy the head position.
			if !yaml_parser_stale_simple_keys(parser) {
				return false
			}

			for i := range parser.simple_keys {
				simple_key := &parser.simple_keys[i]
				if simple_key.possible && simple_key.token_number == parser.tokens_parsed {
					need_more_tokens = true
					break
				}
			}
		}

		// We are finished.
		if !need_more_tokens {
			break
		}
		// Fetch the next token.
		if !yaml_parser_fetch_next_token(parser) {
			return false
		}
	}

	parser.token_available = true
	return true
}

// The dispatcher for token fetchers.
func yaml_parser_fetch_next_token(parser *yaml_parser_t) bool {
	// Ensure that the buffer is initialized.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	// Check if we just started scanning.  Fetch STREAM-START then.
	if !parser.stream_start_produced {
		return yaml_parser_fetch_stream_start(parser)
	}

	// Eat whitespaces and comments until we reach the next token.
	if !yaml_parser_scan_to_next_token(parser) {
		return false
	}

	// Remove obsolete potential simple keys.
	if !yaml_parser_stale_simple_keys(parser) {
		return false
	}

	// Check the indentation level against the current column.
	if !yaml_parser_unroll_indent(parser, parser.mark.column) {
		return false
	}

	// Ensure that the buffer contains at least 4 characters.  4 is the length
	// of the longest indicators ('--- ' and '... ').
	if parser.unread < 4 && !yaml_parser_update_buffer(parser, 4) {
		return false
	}

	// Is it the end of the stream?
	if is_z(parser.buffer, parser.buffer_pos) {
		return yaml_parser_fetch_stream_end(parser)
	}

	// Is it a directive?
	if parser.mark.column == 0 && parser.buffer[parser.buffer_pos] == '%' {
		return yaml_parser_fetch_directive(parser)
	}

	buf := parser.buffer
	pos := parser.buffer_pos

	// Is it the document start indicator?
	if parser.mark.column == 0 && buf[pos] == '-' && buf[pos+1] == '-' && buf[pos+2] == '-' && is_blankz(buf, pos+3) {
		return yaml_parser_fetch_document_indicator(parser, yaml_DOCUMENT_START_TOKEN)
	}

	// Is it the document end indicator?
	if parser.mark.column == 0 && buf[pos] == '.' && buf[pos+1] == '.' && buf[pos+2] == '.' && is_blankz(buf, pos+3) {
		return yaml_parser_fetch_document_indicator(parser, yaml_DOCUMENT_END_TOKEN)
	}

	// Is it the flow sequence start indicator?
	if buf[pos] == '[' {
		return yaml_parser_fetch_flow_collection_start(parser, yaml_FLOW_SEQUENCE_START_TOKEN)
	}

	// Is it the flow mapping start indicator?
	if parser.buffer[parser.buffer_pos] == '{' {
		return yaml_parser_fetch_flow_collection_start(parser, yaml_FLOW_MAPPING_START_TOKEN)
	}

	// Is it the flow sequence end indicator?
	if parser.buffer[parser.buffer_pos] == ']' {
		return yaml_parser_fetch_flow_collection_end(parser,
			yaml_FLOW_SEQUENCE_END_TOKEN)
	}

	// Is it the flow mapping end indicator?
	if parser.buffer[parser.buffer_pos] == '}' {
		return yaml_parser_fetch_flow_collection_end(parser,
			yaml_FLOW_MAPPING_END_TOKEN)
	}

	// Is it the flow entry indicator?
	if parser.buffer[parser.buffer_pos] == ',' {
		return yaml_parser_fetch_flow_entry(parser)
	}

	// Is it the block entry indicator?
	if parser.buffer[parser.buffer_pos] == '-' && is_blankz(parser.buffer, parser.buffer_pos+1) {
		return yaml_parser_fetch_block_entry(parser)
	}

	// Is it the key indicator?
	if parser.buffer[parser.buffer_pos] == '?' && (parser.flow_level > 0 || is_blankz(parser.buffer, parser.buffer_pos+1)) {
		return yaml_parser_fetch_key(parser)
	}

	// Is it the value indicator?
	if parser.buffer[parser.buffer_pos] == ':' && (parser.flow_level > 0 || is_blankz(parser.buffer, parser.buffer_pos+1)) {
		return yaml_parser_fetch_value(parser)
	}

	// Is it an alias?
	if parser.buffer[parser.buffer_pos] == '*' {
		return yaml_parser_fetch_anchor(parser, yaml_ALIAS_TOKEN)
	}

	// Is it an anchor?
	if parser.buffer[parser.buffer_pos] == '&' {
		return yaml_parser_fetch_anchor(parser, yaml_ANCHOR_TOKEN)
	}

	// Is it a tag?
	if parser.buffer[parser.buffer_pos] == '!' {
		return yaml_parser_fetch_tag(parser)
	}

	// Is it a literal scalar?
	if parser.buffer[parser.buffer_pos] == '|' && parser.flow_level == 0 {
		return yaml_parser_fetch_block_scalar(parser, true)
	}

	// Is it a folded scalar?
	if parser.buffer[parser.buffer_pos] == '>' && parser.flow_level == 0 {
		return yaml_parser_fetch_block_scalar(parser, false)
	}

	// Is it a single-quoted scalar?
	if parser.buffer[parser.buffer_pos] == '\'' {
		return yaml_parser_fetch_flow_scalar(parser, true)
	}

	// Is it a double-quoted scalar?
	if parser.buffer[parser.buffer_pos] == '"' {
		return yaml_parser_fetch_flow_scalar(parser, false)
	}

	// Is it a plain scalar?
	//
	// A plain scalar may start with any non-blank characters except
	//
	//      '-', '?', ':', ',', '[', ']', '{', '}',
	//      '#', '&', '*', '!', '|', '>', '\'', '\"',
	//      '%', '@', '`'.
	//
	// In the block context (and, for the '-' indicator, in the flow context
	// too), it may also start with the characters
	//
	//      '-', '?', ':'
	//
	// if it is followed by a non-space character.
	//
	// The last rule is more restrictive than the specification requires.
	// [Go] Make this logic more reasonable.
	//switch parser.buffer[parser.buffer_pos] {
	//case '-', '?', ':', ',', '?', '-', ',', ':', ']', '[', '}', '{', '&', '#', '!', '*', '>', '|', '"', '\'', '@', '%', '-', '`':
	//}
	if !(is_blankz(parser.buffer, parser.buffer_pos) || parser.buffer[parser.buffer_pos] == '-' ||
		parser.buffer[parser.buffer_pos] == '?' || parser.buffer[parser.buffer_pos] == ':' ||
		parser.buffer[parser.buffer_pos] == ',' || parser.buffer[parser.buffer_pos] == '[' ||
		parser.buffer[parser.buffer_pos] == ']' || parser.buffer[parser.buffer_pos] == '{' ||
		parser.buffer[parser.buffer_pos] == '}' || parser.buffer[parser.buffer_pos] == '#' ||
		parser.buffer[parser.buffer_pos] == '&' || parser.buffer[parser.buffer_pos] == '*' ||
		parser.buffer[parser.buffer_pos] == '!' || parser.buffer[parser.buffer_pos] == '|' ||
		parser.buffer[parser.buffer_pos] == '>' || parser.buffer[parser.buffer_pos] == '\'' ||
		parser.buffer[parser.buffer_pos] == '"' || parser.buffer[parser.buffer_pos] == '%' ||
		parser.buffer[parser.buffer_pos] == '@' || parser.buffer[parser.buffer_pos] == '`') ||
		(parser.buffer[parser.buffer_pos] == '-' && !is_blank(parser.buffer, parser.buffer_pos+1)) ||
		(parser.flow_level == 0 &&
			(parser.buffer[parser.buffer_pos] == '?' || parser.buffer[parser.buffer_pos] == ':') &&
			!is_blankz(parser.buffer, parser.buffer_pos+1)) {
		return yaml_parser_fetch_plain_scalar(parser)
	}

	// If we don't determine the token type so far, it is an error.
	return yaml_parser_set_scanner_error(parser,
		"while scanning for the next token", parser.mark,
		"found character that cannot start any token")
}

// Check the list of potential simple keys and remove the positions that
// cannot contain simple keys anymore.
func yaml_parser_stale_simple_keys(parser *yaml_parser_t) bool {
	// Check for a potential simple key for each flow level.
	for i := range parser.simple_keys {
		simple_key := &parser.simple_keys[i]

		// The specification requires that a simple key
		//
		//  - is limited to a single line,
		//  - is shorter than 1024 characters.
		if simple_key.possible && (simple_key.mark.line < parser.mark.line || simple_key.mark.index+1024 < parser.mark.index) {

			// Check if the potential simple key to be removed is required.
			if simple_key.required {
				return yaml_parser_set_scanner_error(parser,
					"while scanning a simple key", simple_key.mark,
					"could not find expected ':'")
			}
			simple_key.possible = false
		}
	}
	return true
}

// Check if a simple key may start at the current position and add it if
// needed.
func yaml_parser_save_simple_key(parser *yaml_parser_t) bool {
	// A simple key is required at the current position if the scanner is in
	// the block context and the current column coincides with the indentation
	// level.

	required := parser.flow_level == 0 && parser.indent == parser.mark.column

	// A simple key is required only when it is the first token in the current
	// line.  Therefore it is always allowed.  But we add a check anyway.
	if required && !parser.simple_key_allowed {
		panic("should not happen")
	}

	//
	// If the current position may start a simple key, save it.
	//
	if parser.simple_key_allowed {
		simple_key := yaml_simple_key_t{
			possible:     true,
			required:     required,
			token_number: parser.tokens_parsed + (len(parser.tokens) - parser.tokens_head),
		}
		simple_key.mark = parser.mark

		if !yaml_parser_remove_simple_key(parser) {
			return false
		}
		parser.simple_keys[len(parser.simple_keys)-1] = simple_key
	}
	return true
}

// Remove a potential simple key at the current flow level.
func yaml_parser_remove_simple_key(parser *yaml_parser_t) bool {
	i := len(parser.simple_keys) - 1
	if parser.simple_keys[i].possible {
		// If the key is required, it is an error.
		if parser.simple_keys[i].required {
			return yaml_parser_set_scanner_error(parser,
				"while scanning a simple key", parser.simple_keys[i].mark,
				"could not find expected ':'")
		}
	}
	// Remove the key from the stack.
	parser.simple_keys[i].possible = false
	return true
}

// Increase the flow level and resize the simple key list if needed.
func yaml_parser_increase_flow_level(parser *yaml_parser_t) bool {
	// Reset the simple key on the next level.
	parser.simple_keys = append(parser.simple_keys, yaml_simple_key_t{})

	// Increase the flow level.
	parser.flow_level++
	return true
}

// Decrease the flow level.
func yaml_parser_decrease_flow_level(parser *yaml_parser_t) bool {
	if parser.flow_level > 0 {
		parser.flow_level--
		parser.simple_keys = parser.simple_keys[:len(parser.simple_keys)-1]
	}
	return true
}

// Push the current indentation level to the stack and set the new level
// the current column is greater than the indentation level.  In this case,
// append or insert the specified token into the token queue.
func yaml_parser_roll_indent(parser *yaml_parser_t, column, number int, typ yaml_token_type_t, mark yaml_mark_t) bool {
	// In the flow context, do nothing.
	if parser.flow_level > 0 {
		return true
	}

	if parser.indent < column {
		// Push the current indentation level to the stack and set the new
		// indentation level.
		parser.indents = append(parser.indents, parser.indent)
		parser.indent = column

		// Create a token and insert it into the queue.
		token := yaml_token_t{
			typ:        typ,
			start_mark: mark,
			end_mark:   mark,
		}
		if number > -1 {
			number -= parser.tokens_parsed
		}
		yaml_insert_token(parser, number, &token)
	}
	return true
}

// Pop indentation levels from the indents stack until the current level
// becomes less or equal to the column.  For each indentation level, append
// the BLOCK-END token.
func yaml_parser_unroll_indent(parser *yaml_parser_t, column int) bool {
	// In the flow context, do nothing.
	if parser.flow_level > 0 {
		return true
	}

	// Loop through the indentation levels in the stack.
	for parser.indent > column {
		// Create a token and append it to the queue.
		token := yaml_token_t{
			typ:        yaml_BLOCK_END_TOKEN,
			start_mark: parser.mark,
			end_mark:   parser.mark,
		}
		yaml_insert_token(parser, -1, &token)

		// Pop the indentation level.
		parser.indent = parser.indents[len(parser.indents)-1]
		parser.indents = parser.indents[:len(parser.indents)-1]
	}
	return true
}

// Initialize the scanner and produce the STREAM-START token.
func yaml_parser_fetch_stream_start(parser *yaml_parser_t) bool {

	// Set the initial indentation.
	parser.indent = -1

	// Initialize the simple key stack.
	parser.simple_keys = append(parser.simple_keys, yaml_simple_key_t{})

	// A simple key is allowed at the beginning of the stream.
	parser.simple_key_allowed = true

	// We have started.
	parser.stream_start_produced = true

	// Create the STREAM-START token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_STREAM_START_TOKEN,
		start_mark: parser.mark,
		end_mark:   parser.mark,
		encoding:   parser.encoding,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the STREAM-END token and shut down the scanner.
func yaml_parser_fetch_stream_end(parser *yaml_parser_t) bool {

	// Force new line.
	if parser.mark.column != 0 {
		parser.mark.column = 0
		parser.mark.line++
	}

	// Reset the indentation level.
	if !yaml_parser_unroll_indent(parser, -1) {
		return false
	}

	// Reset simple keys.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	parser.simple_key_allowed = false

	// Create the STREAM-END token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_STREAM_END_TOKEN,
		start_mark: parser.mark,
		end_mark:   parser.mark,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce a VERSION-DIRECTIVE or TAG-DIRECTIVE token.
func yaml_parser_fetch_directive(parser *yaml_parser_t) bool {
	// Reset the indentation level.
	if !yaml_parser_unroll_indent(parser, -1) {
		return false
	}

	// Reset simple keys.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	parser.simple_key_allowed = false

	// Create the YAML-DIRECTIVE or TAG-DIRECTIVE token.
	token := yaml_token_t{}
	if !yaml_parser_scan_directive(parser, &token) {
		return false
	}
	// Append the token to the queue.
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the DOCUMENT-START or DOCUMENT-END token.
func yaml_parser_fetch_document_indicator(parser *yaml_parser_t, typ yaml_token_type_t) bool {
	// Reset the indentation level.
	if !yaml_parser_unroll_indent(parser, -1) {
		return false
	}

	// Reset simple keys.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	parser.simple_key_allowed = false

	// Consume the token.
	start_mark := parser.mark

	skip(parser)
	skip(parser)
	skip(parser)

	end_mark := parser.mark

	// Create the DOCUMENT-START or DOCUMENT-END token.
	token := yaml_token_t{
		typ:        typ,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	// Append the token to the queue.
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the FLOW-SEQUENCE-START or FLOW-MAPPING-START token.
func yaml_parser_fetch_flow_collection_start(parser *yaml_parser_t, typ yaml_token_type_t) bool {
	// The indicators '[' and '{' may start a simple key.
	if !yaml_parser_save_simple_key(parser) {
		return false
	}

	// Increase the flow level.
	if !yaml_parser_increase_flow_level(parser) {
		return false
	}

	// A simple key may follow the indicators '[' and '{'.
	parser.simple_key_allowed = true

	// Consume the token.
	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the FLOW-SEQUENCE-START of FLOW-MAPPING-START token.
	token := yaml_token_t{
		typ:        typ,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	// Append the token to the queue.
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the FLOW-SEQUENCE-END or FLOW-MAPPING-END token.
func yaml_parser_fetch_flow_collection_end(parser *yaml_parser_t, typ yaml_token_type_t) bool {
	// Reset any potential simple key on the current flow level.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	// Decrease the flow level.
	if !yaml_parser_decrease_flow_level(parser) {
		return false
	}

	// No simple keys after the indicators ']' and '}'.
	parser.simple_key_allowed = false

	// Consume the token.

	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the FLOW-SEQUENCE-END of FLOW-MAPPING-END token.
	token := yaml_token_t{
		typ:        typ,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	// Append the token to the queue.
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the FLOW-ENTRY token.
func yaml_parser_fetch_flow_entry(parser *yaml_parser_t) bool {
	// Reset any potential simple keys on the current flow level.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	// Simple keys are allowed after ','.
	parser.simple_key_allowed = true

	// Consume the token.
	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the FLOW-ENTRY token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_FLOW_ENTRY_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the BLOCK-ENTRY token.
func yaml_parser_fetch_block_entry(parser *yaml_parser_t) bool {
	// Check if the scanner is in the block context.
	if parser.flow_level == 0 {
		// Check if we are allowed to start a new entry.
		if !parser.simple_key_allowed {
			return yaml_parser_set_scanner_error(parser, "", parser.mark,
				"block sequence entries are not allowed in this context")
		}
		// Add the BLOCK-SEQUENCE-START token if needed.
		if !yaml_parser_roll_indent(parser, parser.mark.column, -1, yaml_BLOCK_SEQUENCE_START_TOKEN, parser.mark) {
			return false
		}
	} else {
		// It is an error for the '-' indicator to occur in the flow context,
		// but we let the Parser detect and report about it because the Parser
		// is able to point to the context.
	}

	// Reset any potential simple keys on the current flow level.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	// Simple keys are allowed after '-'.
	parser.simple_key_allowed = true

	// Consume the token.
	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the BLOCK-ENTRY token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_BLOCK_ENTRY_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the KEY token.
func yaml_parser_fetch_key(parser *yaml_parser_t) bool {

	// In the block context, additional checks are required.
	if parser.flow_level == 0 {
		// Check if we are allowed to start a new key (not nessesary simple).
		if !parser.simple_key_allowed {
			return yaml_parser_set_scanner_error(parser, "", parser.mark,
				"mapping keys are not allowed in this context")
		}
		// Add the BLOCK-MAPPING-START token if needed.
		if !yaml_parser_roll_indent(parser, parser.mark.column, -1, yaml_BLOCK_MAPPING_START_TOKEN, parser.mark) {
			return false
		}
	}

	// Reset any potential simple keys on the current flow level.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	// Simple keys are allowed after '?' in the block context.
	parser.simple_key_allowed = parser.flow_level == 0

	// Consume the token.
	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the KEY token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_KEY_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the VALUE token.
func yaml_parser_fetch_value(parser *yaml_parser_t) bool {

	simple_key := &parser.simple_keys[len(parser.simple_keys)-1]

	// Have we found a simple key?
	if simple_key.possible {
		// Create the KEY token and insert it into the queue.
		token := yaml_token_t{
			typ:        yaml_KEY_TOKEN,
			start_mark: simple_key.mark,
			end_mark:   simple_key.mark,
		}
		yaml_insert_token(parser, simple_key.token_number-parser.tokens_parsed, &token)

		// In the block context, we may need to add the BLOCK-MAPPING-START token.
		if !yaml_parser_roll_indent(parser, simple_key.mark.column,
			simple_key.token_number,
			yaml_BLOCK_MAPPING_START_TOKEN, simple_key.mark) {
			return false
		}

		// Remove the simple key.
		simple_key.possible = false

		// A simple key cannot follow another simple key.
		parser.simple_key_allowed = false

	} else {
		// The ':' indicator follows a complex key.

		// In the block context, extra checks are required.
		if parser.flow_level == 0 {

			// Check if we are allowed to start a complex value.
			if !parser.simple_key_allowed {
				return yaml_parser_set_scanner_error(parser, "", parser.mark,
					"mapping values are not allowed in this context")
			}

			// Add the BLOCK-MAPPING-START token if needed.
			if !yaml_parser_roll_indent(parser, parser.mark.column, -1, yaml_BLOCK_MAPPING_START_TOKEN, parser.mark) {
				return false
			}
		}

		// Simple keys after ':' are allowed in the block context.
		parser.simple_key_allowed = parser.flow_level == 0
	}

	// Consume the token.
	start_mark := parser.mark
	skip(parser)
	end_mark := parser.mark

	// Create the VALUE token and append it to the queue.
	token := yaml_token_t{
		typ:        yaml_VALUE_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the ALIAS or ANCHOR token.
func yaml_parser_fetch_anchor(parser *yaml_parser_t, typ yaml_token_type_t) bool {
	// An anchor or an alias could be a simple key.
	if !yaml_parser_save_simple_key(parser) {
		return false
	}

	// A simple key cannot follow an anchor or an alias.
	parser.simple_key_allowed = false

	// Create the ALIAS or ANCHOR token and append it to the queue.
	var token yaml_token_t
	if !yaml_parser_scan_anchor(parser, &token, typ) {
		return false
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the TAG token.
func yaml_parser_fetch_tag(parser *yaml_parser_t) bool {
	// A tag could be a simple key.
	if !yaml_parser_save_simple_key(parser) {
		return false
	}

	// A simple key cannot follow a tag.
	parser.simple_key_allowed = false

	// Create the TAG token and append it to the queue.
	var token yaml_token_t
	if !yaml_parser_scan_tag(parser, &token) {
		return false
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the SCALAR(...,literal) or SCALAR(...,folded) tokens.
func yaml_parser_fetch_block_scalar(parser *yaml_parser_t, literal bool) bool {
	// Remove any potential simple keys.
	if !yaml_parser_remove_simple_key(parser) {
		return false
	}

	// A simple key may follow a block scalar.
	parser.simple_key_allowed = true

	// Create the SCALAR token and append it to the queue.
	var token yaml_token_t
	if !yaml_parser_scan_block_scalar(parser, &token, literal) {
		return false
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the SCALAR(...,single-quoted) or SCALAR(...,double-quoted) tokens.
func yaml_parser_fetch_flow_scalar(parser *yaml_parser_t, single bool) bool {
	// A plain scalar could be a simple key.
	if !yaml_parser_save_simple_key(parser) {
		return false
	}

	// A simple key cannot follow a flow scalar.
	parser.simple_key_allowed = false

	// Create the SCALAR token and append it to the queue.
	var token yaml_token_t
	if !yaml_parser_scan_flow_scalar(parser, &token, single) {
		return false
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Produce the SCALAR(...,plain) token.
func yaml_parser_fetch_plain_scalar(parser *yaml_parser_t) bool {
	// A plain scalar could be a simple key.
	if !yaml_parser_save_simple_key(parser) {
		return false
	}

	// A simple key cannot follow a flow scalar.
	parser.simple_key_allowed = false

	// Create the SCALAR token and append it to the queue.
	var token yaml_token_t
	if !yaml_parser_scan_plain_scalar(parser, &token) {
		return false
	}
	yaml_insert_token(parser, -1, &token)
	return true
}

// Eat whitespaces and comments until the next token is found.
func yaml_parser_scan_to_next_token(parser *yaml_parser_t) bool {

	// Until the next token is not found.
	for {
		// Allow the BOM mark to start a line.
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
		if parser.mark.column == 0 && is_bom(parser.buffer, parser.buffer_pos) {
			skip(parser)
		}

		// Eat whitespaces.
		// Tabs are allowed:
		//  - in the flow context
		//  - in the block context, but not at the beginning of the line or
		//  after '-', '?', or ':' (complex value).
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}

		for parser.buffer[parser.buffer_pos] == ' ' || ((parser.flow_level > 0 || !parser.simple_key_allowed) && parser.buffer[parser.buffer_pos] == '\t') {
			skip(parser)
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}

		// Eat a comment until a line break.
		if parser.buffer[parser.buffer_pos] == '#' {
			for !is_breakz(parser.buffer, parser.buffer_pos) {
				skip(parser)
				if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
					return false
				}
			}
		}

		// If it is a line break, eat it.
		if is_break(parser.buffer, parser.buffer_pos) {
			if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
				return false
			}
			skip_line(parser)

			// In the block context, a new line may start a simple key.
			if parser.flow_level == 0 {
				parser.simple_key_allowed = true
			}
		} else {
			break // We have found a token.
		}
	}

	return true
}

// Scan a YAML-DIRECTIVE or TAG-DIRECTIVE token.
//
// Scope:
//      %YAML    1.1    # a comment \n
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//      %TAG    !yaml!  tag:yaml.org,2002:  \n
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
func yaml_parser_scan_directive(parser *yaml_parser_t, token *yaml_token_t) bool {
	// Eat '%'.
	start_mark := parser.mark
	skip(parser)

	// Scan the directive name.
	var name []byte
	if !yaml_parser_scan_directive_name(parser, start_mark, &name) {
		return false
	}

	// Is it a YAML directive?
	if bytes.Equal(name, []byte("YAML")) {
		// Scan the VERSION directive value.
		var major, minor int8
		if !yaml_parser_scan_version_directive_value(parser, start_mark, &major, &minor) {
			return false
		}
		end_mark := parser.mark

		// Create a VERSION-DIRECTIVE token.
		*token = yaml_token_t{
			typ:        yaml_VERSION_DIRECTIVE_TOKEN,
			start_mark: start_mark,
			end_mark:   end_mark,
			major:      major,
			minor:      minor,
		}

		// Is it a TAG directive?
	} else if bytes.Equal(name, []byte("TAG")) {
		// Scan the TAG directive value.
		var handle, prefix []byte
		if !yaml_parser_scan_tag_directive_value(parser, start_mark, &handle, &prefix) {
			return false
		}
		end_mark := parser.mark

		// Create a TAG-DIRECTIVE token.
		*token = yaml_token_t{
			typ:        yaml_TAG_DIRECTIVE_TOKEN,
			start_mark: start_mark,
			end_mark:   end_mark,
			value:      handle,
			prefix:     prefix,
		}

		// Unknown directive.
	} else {
		yaml_parser_set_scanner_error(parser, "while scanning a directive",
			start_mark, "found unknown directive name")
		return false
	}

	// Eat the rest of the line including any comments.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	for is_blank(parser.buffer, parser.buffer_pos) {
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	if parser.buffer[parser.buffer_pos] == '#' {
		for !is_breakz(parser.buffer, parser.buffer_pos) {
			skip(parser)
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}
	}

	// Check if we are at the end of the line.
	if !is_breakz(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a directive",
			start_mark, "did not find expected comment or line break")
		return false
	}

	// Eat a line break.
	if is_break(parser.buffer, parser.buffer_pos) {
		if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
			return false
		}
		skip_line(parser)
	}

	return true
}

// Scan the directive name.
//
// Scope:
//      %YAML   1.1     # a comment \n
//       ^^^^
//      %TAG    !yaml!  tag:yaml.org,2002:  \n
//       ^^^
//
func yaml_parser_scan_directive_name(parser *yaml_parser_t, start_mark yaml_mark_t, name *[]byte) bool {
	// Consume the directive name.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	var s []byte
	for is_alpha(parser.buffer, parser.buffer_pos) {
		s = read(parser, s)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Check if the name is empty.
	if len(s) == 0 {
		yaml_parser_set_scanner_error(parser, "while scanning a directive",
			start_mark, "could not find expected directive name")
		return false
	}

	// Check for an blank character after the name.
	if !is_blankz(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a directive",
			start_mark, "found unexpected non-alphabetical character")
		return false
	}
	*name = s
	return true
}

// Scan the value of VERSION-DIRECTIVE.
//
// Scope:
//      %YAML   1.1     # a comment \n
//           ^^^^^^
func yaml_parser_scan_version_directive_value(parser *yaml_parser_t, start_mark yaml_mark_t, major, minor *int8) bool {
	// Eat whitespaces.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	for is_blank(parser.buffer, parser.buffer_pos) {
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Consume the major version number.
	if !yaml_parser_scan_version_directive_number(parser, start_mark, major) {
		return false
	}

	// Eat '.'.
	if parser.buffer[parser.buffer_pos] != '.' {
		return yaml_parser_set_scanner_error(parser, "while scanning a %YAML directive",
			start_mark, "did not find expected digit or '.' character")
	}

	skip(parser)

	// Consume the minor version number.
	if !yaml_parser_scan_version_directive_number(parser, start_mark, minor) {
		return false
	}
	return true
}

const max_number_length = 2

// Scan the version number of VERSION-DIRECTIVE.
//
// Scope:
//      %YAML   1.1     # a comment \n
//              ^
//      %YAML   1.1     # a comment \n
//                ^
func yaml_parser_scan_version_directive_number(parser *yaml_parser_t, start_mark yaml_mark_t, number *int8) bool {

	// Repeat while the next character is digit.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	var value, length int8
	for is_digit(parser.buffer, parser.buffer_pos) {
		// Check if the number is too long.
		length++
		if length > max_number_length {
			return yaml_parser_set_scanner_error(parser, "while scanning a %YAML directive",
				start_mark, "found extremely long version number")
		}
		value = value*10 + int8(as_digit(parser.buffer, parser.buffer_pos))
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Check if the number was present.
	if length == 0 {
		return yaml_parser_set_scanner_error(parser, "while scanning a %YAML directive",
			start_mark, "did not find expected version number")
	}
	*number = value
	return true
}

// Scan the value of a TAG-DIRECTIVE token.
//
// Scope:
//      %TAG    !yaml!  tag:yaml.org,2002:  \n
//          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
func yaml_parser_scan_tag_directive_value(parser *yaml_parser_t, start_mark yaml_mark_t, handle, prefix *[]byte) bool {
	var handle_value, prefix_value []byte

	// Eat whitespaces.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	for is_blank(parser.buffer, parser.buffer_pos) {
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Scan a handle.
	if !yaml_parser_scan_tag_handle(parser, true, start_mark, &handle_value) {
		return false
	}

	// Expect a whitespace.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	if !is_blank(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a %TAG directive",
			start_mark, "did not find expected whitespace")
		return false
	}

	// Eat whitespaces.
	for is_blank(parser.buffer, parser.buffer_pos) {
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Scan a prefix.
	if !yaml_parser_scan_tag_uri(parser, true, nil, start_mark, &prefix_value) {
		return false
	}

	// Expect a whitespace or line break.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	if !is_blankz(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a %TAG directive",
			start_mark, "did not find expected whitespace or line break")
		return false
	}

	*handle = handle_value
	*prefix = prefix_value
	return true
}

func yaml_parser_scan_anchor(parser *yaml_parser_t, token *yaml_token_t, typ yaml_token_type_t) bool {
	var s []byte

	// Eat the indicator character.
	start_mark := parser.mark
	skip(parser)

	// Consume the value.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	for is_alpha(parser.buffer, parser.buffer_pos) {
		s = read(parser, s)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	end_mark := parser.mark

	/*
	 * Check if length of the anchor is greater than 0 and it is followed by
	 * a whitespace character or one of the indicators:
	 *
	 *      '?', ':', ',', ']', '}', '%', '@', '`'.
	 */

	if len(s) == 0 ||
		!(is_blankz(parser.buffer, parser.buffer_pos) || parser.buffer[parser.buffer_pos] == '?' ||
			parser.buffer[parser.buffer_pos] == ':' || parser.buffer[parser.buffer_pos] == ',' ||
			parser.buffer[parser.buffer_pos] == ']' || parser.buffer[parser.buffer_pos] == '}' ||
			parser.buffer[parser.buffer_pos] == '%' || parser.buffer[parser.buffer_pos] == '@' ||
			parser.buffer[parser.buffer_pos] == '`') {
		context := "while scanning an alias"
		if typ == yaml_ANCHOR_TOKEN {
			context = "while scanning an anchor"
		}
		yaml_parser_set_scanner_error(parser, context, start_mark,
			"did not find expected alphabetic or numeric character")
		return false
	}

	// Create a token.
	*token = yaml_token_t{
		typ:        typ,
		start_mark: start_mark,
		end_mark:   end_mark,
		value:      s,
	}

	return true
}

/*
 * Scan a TAG token.
 */

func yaml_parser_scan_tag(parser *yaml_parser_t, token *yaml_token_t) bool {
	var handle, suffix []byte

	start_mark := parser.mark

	// Check if the tag is in the canonical form.
	if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
		return false
	}

	if parser.buffer[parser.buffer_pos+1] == '<' {
		// Keep the handle as ''

		// Eat '!<'
		skip(parser)
		skip(parser)

		// Consume the tag value.
		if !yaml_parser_scan_tag_uri(parser, false, nil, start_mark, &suffix) {
			return false
		}

		// Check for '>' and eat it.
		if parser.buffer[parser.buffer_pos] != '>' {
			yaml_parser_set_scanner_error(parser, "while scanning a tag",
				start_mark, "did not find the expected '>'")
			return false
		}

		skip(parser)
	} else {
		// The tag has either the '!suffix' or the '!handle!suffix' form.

		// First, try to scan a handle.
		if !yaml_parser_scan_tag_handle(parser, false, start_mark, &handle) {
			return false
		}

		// Check if it is, indeed, handle.
		if handle[0] == '!' && len(handle) > 1 && handle[len(handle)-1] == '!' {
			// Scan the suffix now.
			if !yaml_parser_scan_tag_uri(parser, false, nil, start_mark, &suffix) {
				return false
			}
		} else {
			// It wasn't a handle after all.  Scan the rest of the tag.
			if !yaml_parser_scan_tag_uri(parser, false, handle, start_mark, &suffix) {
				return false
			}

			// Set the handle to '!'.
			handle = []byte{'!'}

			// A special case: the '!' tag.  Set the handle to '' and the
			// suffix to '!'.
			if len(suffix) == 0 {
				handle, suffix = suffix, handle
			}
		}
	}

	// Check the character which ends the tag.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	if !is_blankz(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a tag",
			start_mark, "did not find expected whitespace or line break")
		return false
	}

	end_mark := parser.mark

	// Create a token.
	*token = yaml_token_t{
		typ:        yaml_TAG_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
		value:      handle,
		suffix:     suffix,
	}
	return true
}

// Scan a tag handle.
func yaml_parser_scan_tag_handle(parser *yaml_parser_t, directive bool, start_mark yaml_mark_t, handle *[]byte) bool {
	// Check the initial '!' character.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	if parser.buffer[parser.buffer_pos] != '!' {
		yaml_parser_set_scanner_tag_error(parser, directive,
			start_mark, "did not find expected '!'")
		return false
	}

	var s []byte

	// Copy the '!' character.
	s = read(parser, s)

	// Copy all subsequent alphabetical and numerical characters.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	for is_alpha(parser.buffer, parser.buffer_pos) {
		s = read(parser, s)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Check if the trailing character is '!' and copy it.
	if parser.buffer[parser.buffer_pos] == '!' {
		s = read(parser, s)
	} else {
		// It's either the '!' tag or not really a tag handle.  If it's a %TAG
		// directive, it's an error.  If it's a tag token, it must be a part of URI.
		if directive && !(s[0] == '!' && s[1] == 0) {
			yaml_parser_set_scanner_tag_error(parser, directive,
				start_mark, "did not find expected '!'")
			return false
		}
	}

	*handle = s
	return true
}

// Scan a tag.
func yaml_parser_scan_tag_uri(parser *yaml_parser_t, directive bool, head []byte, start_mark yaml_mark_t, uri *[]byte) bool {
	//size_t length = head ? strlen((char *)head) : 0
	var s []byte
	length := len(head)

	// Copy the head if needed.
	//
	// Note that we don't copy the leading '!' character.
	if length > 0 {
		s = append(s, head[1:]...)
	}

	// Scan the tag.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	// The set of characters that may appear in URI is as follows:
	//
	//      '0'-'9', 'A'-'Z', 'a'-'z', '_', '-', ';', '/', '?', ':', '@', '&',
	//      '=', '+', '$', ',', '.', '!', '~', '*', '\'', '(', ')', '[', ']',
	//      '%'.
	// [Go] Convert this into more reasonable logic.
	for is_alpha(parser.buffer, parser.buffer_pos) || parser.buffer[parser.buffer_pos] == ';' ||
		parser.buffer[parser.buffer_pos] == '/' || parser.buffer[parser.buffer_pos] == '?' ||
		parser.buffer[parser.buffer_pos] == ':' || parser.buffer[parser.buffer_pos] == '@' ||
		parser.buffer[parser.buffer_pos] == '&' || parser.buffer[parser.buffer_pos] == '=' ||
		parser.buffer[parser.buffer_pos] == '+' || parser.buffer[parser.buffer_pos] == '$' ||
		parser.buffer[parser.buffer_pos] == ',' || parser.buffer[parser.buffer_pos] == '.' ||
		parser.buffer[parser.buffer_pos] == '!' || parser.buffer[parser.buffer_pos] == '~' ||
		parser.buffer[parser.buffer_pos] == '*' || parser.buffer[parser.buffer_pos] == '\'' ||
		parser.buffer[parser.buffer_pos] == '(' || parser.buffer[parser.buffer_pos] == ')' ||
		parser.buffer[parser.buffer_pos] == '[' || parser.buffer[parser.buffer_pos] == ']' ||
		parser.buffer[parser.buffer_pos] == '%' {
		// Check if it is a URI-escape sequence.
		if parser.buffer[parser.buffer_pos] == '%' {
			if !yaml_parser_scan_uri_escapes(parser, directive, start_mark, &s) {
				return false
			}
		} else {
			s = read(parser, s)
			length++
		}
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}

	// Check if the tag is non-empty.
	if length == 0 {
		yaml_parser_set_scanner_tag_error(parser, directive,
			start_mark, "did not find expected tag URI")
		return false
	}
	*uri = s
	return true
}

// Decode an URI-escape sequence corresponding to a single UTF-8 character.
func yaml_parser_scan_uri_escapes(parser *yaml_parser_t, directive bool, start_mark yaml_mark_t, s *[]byte) bool {

	// Decode the required number of characters.
	w := 1024
	for w > 0 {
		// Check for a URI-escaped octet.
		if parser.unread < 3 && !yaml_parser_update_buffer(parser, 3) {
			return false
		}

		if !(parser.buffer[parser.buffer_pos] == '%' &&
			is_hex(parser.buffer, parser.buffer_pos+1) &&
			is_hex(parser.buffer, parser.buffer_pos+2)) {
			return yaml_parser_set_scanner_tag_error(parser, directive,
				start_mark, "did not find URI escaped octet")
		}

		// Get the octet.
		octet := byte((as_hex(parser.buffer, parser.buffer_pos+1) << 4) + as_hex(parser.buffer, parser.buffer_pos+2))

		// If it is the leading octet, determine the length of the UTF-8 sequence.
		if w == 1024 {
			w = width(octet)
			if w == 0 {
				return yaml_parser_set_scanner_tag_error(parser, directive,
					start_mark, "found an incorrect leading UTF-8 octet")
			}
		} else {
			// Check if the trailing octet is correct.
			if octet&0xC0 != 0x80 {
				return yaml_parser_set_scanner_tag_error(parser, directive,
					start_mark, "found an incorrect trailing UTF-8 octet")
			}
		}

		// Copy the octet and move the pointers.
		*s = append(*s, octet)
		skip(parser)
		skip(parser)
		skip(parser)
		w--
	}
	return true
}

// Scan a block scalar.
func yaml_parser_scan_block_scalar(parser *yaml_parser_t, token *yaml_token_t, literal bool) bool {
	// Eat the indicator '|' or '>'.
	start_mark := parser.mark
	skip(parser)

	// Scan the additional block scalar indicators.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}

	// Check for a chomping indicator.
	var chomping, increment int
	if parser.buffer[parser.buffer_pos] == '+' || parser.buffer[parser.buffer_pos] == '-' {
		// Set the chomping method and eat the indicator.
		if parser.buffer[parser.buffer_pos] == '+' {
			chomping = +1
		} else {
			chomping = -1
		}
		skip(parser)

		// Check for an indentation indicator.
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
		if is_digit(parser.buffer, parser.buffer_pos) {
			// Check that the indentation is greater than 0.
			if parser.buffer[parser.buffer_pos] == '0' {
				yaml_parser_set_scanner_error(parser, "while scanning a block scalar",
					start_mark, "found an indentation indicator equal to 0")
				return false
			}

			// Get the indentation level and eat the indicator.
			increment = as_digit(parser.buffer, parser.buffer_pos)
			skip(parser)
		}

	} else if is_digit(parser.buffer, parser.buffer_pos) {
		// Do the same as above, but in the opposite order.

		if parser.buffer[parser.buffer_pos] == '0' {
			yaml_parser_set_scanner_error(parser, "while scanning a block scalar",
				start_mark, "found an indentation indicator equal to 0")
			return false
		}
		increment = as_digit(parser.buffer, parser.buffer_pos)
		skip(parser)

		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
		if parser.buffer[parser.buffer_pos] == '+' || parser.buffer[parser.buffer_pos] == '-' {
			if parser.buffer[parser.buffer_pos] == '+' {
				chomping = +1
			} else {
				chomping = -1
			}
			skip(parser)
		}
	}

	// Eat whitespaces and comments to the end of the line.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	for is_blank(parser.buffer, parser.buffer_pos) {
		skip(parser)
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
	}
	if parser.buffer[parser.buffer_pos] == '#' {
		for !is_breakz(parser.buffer, parser.buffer_pos) {
			skip(parser)
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}
	}

	// Check if we are at the end of the line.
	if !is_breakz(parser.buffer, parser.buffer_pos) {
		yaml_parser_set_scanner_error(parser, "while scanning a block scalar",
			start_mark, "did not find expected comment or line break")
		return false
	}

	// Eat a line break.
	if is_break(parser.buffer, parser.buffer_pos) {
		if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
			return false
		}
		skip_line(parser)
	}

	end_mark := parser.mark

	// Set the indentation level if it was specified.
	var indent int
	if increment > 0 {
		if parser.indent >= 0 {
			indent = parser.indent + increment
		} else {
			indent = increment
		}
	}

	// Scan the leading line breaks and determine the indentation level if needed.
	var s, leading_break, trailing_breaks []byte
	if !yaml_parser_scan_block_scalar_breaks(parser, &indent, &trailing_breaks, start_mark, &end_mark) {
		return false
	}

	// Scan the block scalar content.
	if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
		return false
	}
	var leading_blank, trailing_blank bool
	for parser.mark.column == indent && !is_z(parser.buffer, parser.buffer_pos) {
		// We are at the beginning of a non-empty line.

		// Is it a trailing whitespace?
		trailing_blank = is_blank(parser.buffer, parser.buffer_pos)

		// Check if we need to fold the leading line break.
		if !literal && !leading_blank && !trailing_blank && len(leading_break) > 0 && leading_break[0] == '\n' {
			// Do we need to join the lines by space?
			if len(trailing_breaks) == 0 {
				s = append(s, ' ')
			}
		} else {
			s = append(s, leading_break...)
		}
		leading_break = leading_break[:0]

		// Append the remaining line breaks.
		s = append(s, trailing_breaks...)
		trailing_breaks = trailing_breaks[:0]

		// Is it a leading whitespace?
		leading_blank = is_blank(parser.buffer, parser.buffer_pos)

		// Consume the current line.
		for !is_breakz(parser.buffer, parser.buffer_pos) {
			s = read(parser, s)
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}

		// Consume the line break.
		if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
			return false
		}

		leading_break = read_line(parser, leading_break)

		// Eat the following indentation spaces and line breaks.
		if !yaml_parser_scan_block_scalar_breaks(parser, &indent, &trailing_breaks, start_mark, &end_mark) {
			return false
		}
	}

	// Chomp the tail.
	if chomping != -1 {
		s = append(s, leading_break...)
	}
	if chomping == 1 {
		s = append(s, trailing_breaks...)
	}

	// Create a token.
	*token = yaml_token_t{
		typ:        yaml_SCALAR_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
		value:      s,
		style:      yaml_LITERAL_SCALAR_STYLE,
	}
	if !literal {
		token.style = yaml_FOLDED_SCALAR_STYLE
	}
	return true
}

// Scan indentation spaces and line breaks for a block scalar.  Determine the
// indentation level if needed.
func yaml_parser_scan_block_scalar_breaks(parser *yaml_parser_t, indent *int, breaks *[]byte, start_mark yaml_mark_t, end_mark *yaml_mark_t) bool {
	*end_mark = parser.mark

	// Eat the indentation spaces and line breaks.
	max_indent := 0
	for {
		// Eat the indentation spaces.
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}
		for (*indent == 0 || parser.mark.column < *indent) && is_space(parser.buffer, parser.buffer_pos) {
			skip(parser)
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}
		if parser.mark.column > max_indent {
			max_indent = parser.mark.column
		}

		// Check for a tab character messing the indentation.
		if (*indent == 0 || parser.mark.column < *indent) && is_tab(parser.buffer, parser.buffer_pos) {
			return yaml_parser_set_scanner_error(parser, "while scanning a block scalar",
				start_mark, "found a tab character where an indentation space is expected")
		}

		// Have we found a non-empty line?
		if !is_break(parser.buffer, parser.buffer_pos) {
			break
		}

		// Consume the line break.
		if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
			return false
		}
		// [Go] Should really be returning breaks instead.
		*breaks = read_line(parser, *breaks)
		*end_mark = parser.mark
	}

	// Determine the indentation level if needed.
	if *indent == 0 {
		*indent = max_indent
		if *indent < parser.indent+1 {
			*indent = parser.indent + 1
		}
		if *indent < 1 {
			*indent = 1
		}
	}
	return true
}

// Scan a quoted scalar.
func yaml_parser_scan_flow_scalar(parser *yaml_parser_t, token *yaml_token_t, single bool) bool {
	// Eat the left quote.
	start_mark := parser.mark
	skip(parser)

	// Consume the content of the quoted scalar.
	var s, leading_break, trailing_breaks, whitespaces []byte
	for {
		// Check that there are no document indicators at the beginning of the line.
		if parser.unread < 4 && !yaml_parser_update_buffer(parser, 4) {
			return false
		}

		if parser.mark.column == 0 &&
			((parser.buffer[parser.buffer_pos+0] == '-' &&
				parser.buffer[parser.buffer_pos+1] == '-' &&
				parser.buffer[parser.buffer_pos+2] == '-') ||
				(parser.buffer[parser.buffer_pos+0] == '.' &&
					parser.buffer[parser.buffer_pos+1] == '.' &&
					parser.buffer[parser.buffer_pos+2] == '.')) &&
			is_blankz(parser.buffer, parser.buffer_pos+3) {
			yaml_parser_set_scanner_error(parser, "while scanning a quoted scalar",
				start_mark, "found unexpected document indicator")
			return false
		}

		// Check for EOF.
		if is_z(parser.buffer, parser.buffer_pos) {
			yaml_parser_set_scanner_error(parser, "while scanning a quoted scalar",
				start_mark, "found unexpected end of stream")
			return false
		}

		// Consume non-blank characters.
		leading_blanks := false
		for !is_blankz(parser.buffer, parser.buffer_pos) {
			if single && parser.buffer[parser.buffer_pos] == '\'' && parser.buffer[parser.buffer_pos+1] == '\'' {
				// Is is an escaped single quote.
				s = append(s, '\'')
				skip(parser)
				skip(parser)

			} else if single && parser.buffer[parser.buffer_pos] == '\'' {
				// It is a right single quote.
				break
			} else if !single && parser.buffer[parser.buffer_pos] == '"' {
				// It is a right double quote.
				break

			} else if !single && parser.buffer[parser.buffer_pos] == '\\' && is_break(parser.buffer, parser.buffer_pos+1) {
				// It is an escaped line break.
				if parser.unread < 3 && !yaml_parser_update_buffer(parser, 3) {
					return false
				}
				skip(parser)
				skip_line(parser)
				leading_blanks = true
				break

			} else if !single && parser.buffer[parser.buffer_pos] == '\\' {
				// It is an escape sequence.
				code_length := 0

				// Check the escape character.
				switch parser.buffer[parser.buffer_pos+1] {
				case '0':
					s = append(s, 0)
				case 'a':
					s = append(s, '\x07')
				case 'b':
					s = append(s, '\x08')
				case 't', '\t':
					s = append(s, '\x09')
				case 'n':
					s = append(s, '\x0A')
				case 'v':
					s = append(s, '\x0B')
				case 'f':
					s = append(s, '\x0C')
				case 'r':
					s = append(s, '\x0D')
				case 'e':
					s = append(s, '\x1B')
				case ' ':
					s = append(s, '\x20')
				case '"':
					s = append(s, '"')
				case '\'':
					s = append(s, '\'')
				case '\\':
					s = append(s, '\\')
				case 'N': // NEL (#x85)
					s = append(s, '\xC2')
					s = append(s, '\x85')
				case '_': // #xA0
					s = append(s, '\xC2')
					s = append(s, '\xA0')
				case 'L': // LS (#x2028)
					s = append(s, '\xE2')
					s = append(s, '\x80')
					s = append(s, '\xA8')
				case 'P': // PS (#x2029)
					s = append(s, '\xE2')
					s = append(s, '\x80')
					s = append(s, '\xA9')
				case 'x':
					code_length = 2
				case 'u':
					code_length = 4
				case 'U':
					code_length = 8
				default:
					yaml_parser_set_scanner_error(parser, "while parsing a quoted scalar",
						start_mark, "found unknown escape character")
					return false
				}

				skip(parser)
				skip(parser)

				// Consume an arbitrary escape code.
				if code_length > 0 {
					var value int

					// Scan the character value.
					if parser.unread < code_length && !yaml_parser_update_buffer(parser, code_length) {
						return false
					}
					for k := 0; k < code_length; k++ {
						if !is_hex(parser.buffer, parser.buffer_pos+k) {
							yaml_parser_set_scanner_error(parser, "while parsing a quoted scalar",
								start_mark, "did not find expected hexdecimal number")
							return false
						}
						value = (value << 4) + as_hex(parser.buffer, parser.buffer_pos+k)
					}

					// Check the value and write the character.
					if (value >= 0xD800 && value <= 0xDFFF) || value > 0x10FFFF {
						yaml_parser_set_scanner_error(parser, "while parsing a quoted scalar",
							start_mark, "found invalid Unicode character escape code")
						return false
					}
					if value <= 0x7F {
						s = append(s, byte(value))
					} else if value <= 0x7FF {
						s = append(s, byte(0xC0+(value>>6)))
						s = append(s, byte(0x80+(value&0x3F)))
					} else if value <= 0xFFFF {
						s = append(s, byte(0xE0+(value>>12)))
						s = append(s, byte(0x80+((value>>6)&0x3F)))
						s = append(s, byte(0x80+(value&0x3F)))
					} else {
						s = append(s, byte(0xF0+(value>>18)))
						s = append(s, byte(0x80+((value>>12)&0x3F)))
						s = append(s, byte(0x80+((value>>6)&0x3F)))
						s = append(s, byte(0x80+(value&0x3F)))
					}

					// Advance the pointer.
					for k := 0; k < code_length; k++ {
						skip(parser)
					}
				}
			} else {
				// It is a non-escaped non-blank character.
				s = read(parser, s)
			}
			if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
				return false
			}
		}

		// Check if we are at the end of the scalar.
		if single {
			if parser.buffer[parser.buffer_pos] == '\'' {
				break
			}
		} else {
			if parser.buffer[parser.buffer_pos] == '"' {
				break
			}
		}

		// Consume blank characters.
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}

		for is_blank(parser.buffer, parser.buffer_pos) || is_break(parser.buffer, parser.buffer_pos) {
			if is_blank(parser.buffer, parser.buffer_pos) {
				// Consume a space or a tab character.
				if !leading_blanks {
					whitespaces = read(parser, whitespaces)
				} else {
					skip(parser)
				}
			} else {
				if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
					return false
				}

				// Check if it is a first line break.
				if !leading_blanks {
					whitespaces = whitespaces[:0]
					leading_break = read_line(parser, leading_break)
					leading_blanks = true
				} else {
					trailing_breaks = read_line(parser, trailing_breaks)
				}
			}
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}

		// Join the whitespaces or fold line breaks.
		if leading_blanks {
			// Do we need to fold line breaks?
			if len(leading_break) > 0 && leading_break[0] == '\n' {
				if len(trailing_breaks) == 0 {
					s = append(s, ' ')
				} else {
					s = append(s, trailing_breaks...)
				}
			} else {
				s = append(s, leading_break...)
				s = append(s, trailing_breaks...)
			}
			trailing_breaks = trailing_breaks[:0]
			leading_break = leading_break[:0]
		} else {
			s = append(s, whitespaces...)
			whitespaces = whitespaces[:0]
		}
	}

	// Eat the right quote.
	skip(parser)
	end_mark := parser.mark

	// Create a token.
	*token = yaml_token_t{
		typ:        yaml_SCALAR_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
		value:      s,
		style:      yaml_SINGLE_QUOTED_SCALAR_STYLE,
	}
	if !single {
		token.style = yaml_DOUBLE_QUOTED_SCALAR_STYLE
	}
	return true
}

// Scan a plain scalar.
func yaml_parser_scan_plain_scalar(parser *yaml_parser_t, token *yaml_token_t) bool {

	var s, leading_break, trailing_breaks, whitespaces []byte
	var leading_blanks bool
	var indent = parser.indent + 1

	start_mark := parser.mark
	end_mark := parser.mark

	// Consume the content of the plain scalar.
	for {
		// Check for a document indicator.
		if parser.unread < 4 && !yaml_parser_update_buffer(parser, 4) {
			return false
		}
		if parser.mark.column == 0 &&
			((parser.buffer[parser.buffer_pos+0] == '-' &&
				parser.buffer[parser.buffer_pos+1] == '-' &&
				parser.buffer[parser.buffer_pos+2] == '-') ||
				(parser.buffer[parser.buffer_pos+0] == '.' &&
					parser.buffer[parser.buffer_pos+1] == '.' &&
					parser.buffer[parser.buffer_pos+2] == '.')) &&
			is_blankz(parser.buffer, parser.buffer_pos+3) {
			break
		}

		// Check for a comment.
		if parser.buffer[parser.buffer_pos] == '#' {
			break
		}

		// Consume non-blank characters.
		for !is_blankz(parser.buffer, parser.buffer_pos) {

			// Check for 'x:x' in the flow context. TODO: Fix the test "spec-08-13".
			if parser.flow_level > 0 &&
				parser.buffer[parser.buffer_pos] == ':' &&
				!is_blankz(parser.buffer, parser.buffer_pos+1) {
				yaml_parser_set_scanner_error(parser, "while scanning a plain scalar",
					start_mark, "found unexpected ':'")
				return false
			}

			// Check for indicators that may end a plain scalar.
			if (parser.buffer[parser.buffer_pos] == ':' && is_blankz(parser.buffer, parser.buffer_pos+1)) ||
				(parser.flow_level > 0 &&
					(parser.buffer[parser.buffer_pos] == ',' || parser.buffer[parser.buffer_pos] == ':' ||
						parser.buffer[parser.buffer_pos] == '?' || parser.buffer[parser.buffer_pos] == '[' ||
						parser.buffer[parser.buffer_pos] == ']' || parser.buffer[parser.buffer_pos] == '{' ||
						parser.buffer[parser.buffer_pos] == '}')) {
				break
			}

			// Check if we need to join whitespaces and breaks.
			if leading_blanks || len(whitespaces) > 0 {
				if leading_blanks {
					// Do we need to fold line breaks?
					if leading_break[0] == '\n' {
						if len(trailing_breaks) == 0 {
							s = append(s, ' ')
						} else {
							s = append(s, trailing_breaks...)
						}
					} else {
						s = append(s, leading_break...)
						s = append(s, trailing_breaks...)
					}
					trailing_breaks = trailing_breaks[:0]
					leading_break = leading_break[:0]
					leading_blanks = false
				} else {
					s = append(s, whitespaces...)
					whitespaces = whitespaces[:0]
				}
			}

			// Copy the character.
			s = read(parser, s)

			end_mark = parser.mark
			if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
				return false
			}
		}

		// Is it the end?
		if !(is_blank(parser.buffer, parser.buffer_pos) || is_break(parser.buffer, parser.buffer_pos)) {
			break
		}

		// Consume blank characters.
		if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
			return false
		}

		for is_blank(parser.buffer, parser.buffer_pos) || is_break(parser.buffer, parser.buffer_pos) {
			if is_blank(parser.buffer, parser.buffer_pos) {

				// Check for tab character that abuse indentation.
				if leading_blanks && parser.mark.column < indent && is_tab(parser.buffer, parser.buffer_pos) {
					yaml_parser_set_scanner_error(parser, "while scanning a plain scalar",
						start_mark, "found a tab character that violate indentation")
					return false
				}

				// Consume a space or a tab character.
				if !leading_blanks {
					whitespaces = read(parser, whitespaces)
				} else {
					skip(parser)
				}
			} else {
				if parser.unread < 2 && !yaml_parser_update_buffer(parser, 2) {
					return false
				}

				// Check if it is a first line break.
				if !leading_blanks {
					whitespaces = whitespaces[:0]
					leading_break = read_line(parser, leading_break)
					leading_blanks = true
				} else {
					trailing_breaks = read_line(parser, trailing_breaks)
				}
			}
			if parser.unread < 1 && !yaml_parser_update_buffer(parser, 1) {
				return false
			}
		}

		// Check indentation level.
		if parser.flow_level == 0 && parser.mark.column < indent {
			break
		}
	}

	// Create a token.
	*token = yaml_token_t{
		typ:        yaml_SCALAR_TOKEN,
		start_mark: start_mark,
		end_mark:   end_mark,
		value:      s,
		style:      yaml_PLAIN_SCALAR_STYLE,
	}

	// Note that we change the 'simple_key_allowed' flag.
	if leading_blanks {
		parser.simple_key_allowed = true
	}
	return true
}
