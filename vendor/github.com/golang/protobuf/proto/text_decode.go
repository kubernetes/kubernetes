// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"encoding"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/protobuf/encoding/prototext"
	protoV2 "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

const wrapTextUnmarshalV2 = false

// ParseError is returned by UnmarshalText.
type ParseError struct {
	Message string

	// Deprecated: Do not use.
	Line, Offset int
}

func (e *ParseError) Error() string {
	if wrapTextUnmarshalV2 {
		return e.Message
	}
	if e.Line == 1 {
		return fmt.Sprintf("line 1.%d: %v", e.Offset, e.Message)
	}
	return fmt.Sprintf("line %d: %v", e.Line, e.Message)
}

// UnmarshalText parses a proto text formatted string into m.
func UnmarshalText(s string, m Message) error {
	if u, ok := m.(encoding.TextUnmarshaler); ok {
		return u.UnmarshalText([]byte(s))
	}

	m.Reset()
	mi := MessageV2(m)

	if wrapTextUnmarshalV2 {
		err := prototext.UnmarshalOptions{
			AllowPartial: true,
		}.Unmarshal([]byte(s), mi)
		if err != nil {
			return &ParseError{Message: err.Error()}
		}
		return checkRequiredNotSet(mi)
	} else {
		if err := newTextParser(s).unmarshalMessage(mi.ProtoReflect(), ""); err != nil {
			return err
		}
		return checkRequiredNotSet(mi)
	}
}

type textParser struct {
	s            string // remaining input
	done         bool   // whether the parsing is finished (success or error)
	backed       bool   // whether back() was called
	offset, line int
	cur          token
}

type token struct {
	value    string
	err      *ParseError
	line     int    // line number
	offset   int    // byte number from start of input, not start of line
	unquoted string // the unquoted version of value, if it was a quoted string
}

func newTextParser(s string) *textParser {
	p := new(textParser)
	p.s = s
	p.line = 1
	p.cur.line = 1
	return p
}

func (p *textParser) unmarshalMessage(m protoreflect.Message, terminator string) (err error) {
	md := m.Descriptor()
	fds := md.Fields()

	// A struct is a sequence of "name: value", terminated by one of
	// '>' or '}', or the end of the input.  A name may also be
	// "[extension]" or "[type/url]".
	//
	// The whole struct can also be an expanded Any message, like:
	// [type/url] < ... struct contents ... >
	seen := make(map[protoreflect.FieldNumber]bool)
	for {
		tok := p.next()
		if tok.err != nil {
			return tok.err
		}
		if tok.value == terminator {
			break
		}
		if tok.value == "[" {
			if err := p.unmarshalExtensionOrAny(m, seen); err != nil {
				return err
			}
			continue
		}

		// This is a normal, non-extension field.
		name := protoreflect.Name(tok.value)
		fd := fds.ByName(name)
		switch {
		case fd == nil:
			gd := fds.ByName(protoreflect.Name(strings.ToLower(string(name))))
			if gd != nil && gd.Kind() == protoreflect.GroupKind && gd.Message().Name() == name {
				fd = gd
			}
		case fd.Kind() == protoreflect.GroupKind && fd.Message().Name() != name:
			fd = nil
		case fd.IsWeak() && fd.Message().IsPlaceholder():
			fd = nil
		}
		if fd == nil {
			typeName := string(md.FullName())
			if m, ok := m.Interface().(Message); ok {
				t := reflect.TypeOf(m)
				if t.Kind() == reflect.Ptr {
					typeName = t.Elem().String()
				}
			}
			return p.errorf("unknown field name %q in %v", name, typeName)
		}
		if od := fd.ContainingOneof(); od != nil && m.WhichOneof(od) != nil {
			return p.errorf("field '%s' would overwrite already parsed oneof '%s'", name, od.Name())
		}
		if fd.Cardinality() != protoreflect.Repeated && seen[fd.Number()] {
			return p.errorf("non-repeated field %q was repeated", fd.Name())
		}
		seen[fd.Number()] = true

		// Consume any colon.
		if err := p.checkForColon(fd); err != nil {
			return err
		}

		// Parse into the field.
		v := m.Get(fd)
		if !m.Has(fd) && (fd.IsList() || fd.IsMap() || fd.Message() != nil) {
			v = m.Mutable(fd)
		}
		if v, err = p.unmarshalValue(v, fd); err != nil {
			return err
		}
		m.Set(fd, v)

		if err := p.consumeOptionalSeparator(); err != nil {
			return err
		}
	}
	return nil
}

func (p *textParser) unmarshalExtensionOrAny(m protoreflect.Message, seen map[protoreflect.FieldNumber]bool) error {
	name, err := p.consumeExtensionOrAnyName()
	if err != nil {
		return err
	}

	// If it contains a slash, it's an Any type URL.
	if slashIdx := strings.LastIndex(name, "/"); slashIdx >= 0 {
		tok := p.next()
		if tok.err != nil {
			return tok.err
		}
		// consume an optional colon
		if tok.value == ":" {
			tok = p.next()
			if tok.err != nil {
				return tok.err
			}
		}

		var terminator string
		switch tok.value {
		case "<":
			terminator = ">"
		case "{":
			terminator = "}"
		default:
			return p.errorf("expected '{' or '<', found %q", tok.value)
		}

		mt, err := protoregistry.GlobalTypes.FindMessageByURL(name)
		if err != nil {
			return p.errorf("unrecognized message %q in google.protobuf.Any", name[slashIdx+len("/"):])
		}
		m2 := mt.New()
		if err := p.unmarshalMessage(m2, terminator); err != nil {
			return err
		}
		b, err := protoV2.Marshal(m2.Interface())
		if err != nil {
			return p.errorf("failed to marshal message of type %q: %v", name[slashIdx+len("/"):], err)
		}

		urlFD := m.Descriptor().Fields().ByName("type_url")
		valFD := m.Descriptor().Fields().ByName("value")
		if seen[urlFD.Number()] {
			return p.errorf("Any message unpacked multiple times, or %q already set", urlFD.Name())
		}
		if seen[valFD.Number()] {
			return p.errorf("Any message unpacked multiple times, or %q already set", valFD.Name())
		}
		m.Set(urlFD, protoreflect.ValueOfString(name))
		m.Set(valFD, protoreflect.ValueOfBytes(b))
		seen[urlFD.Number()] = true
		seen[valFD.Number()] = true
		return nil
	}

	xname := protoreflect.FullName(name)
	xt, _ := protoregistry.GlobalTypes.FindExtensionByName(xname)
	if xt == nil && isMessageSet(m.Descriptor()) {
		xt, _ = protoregistry.GlobalTypes.FindExtensionByName(xname.Append("message_set_extension"))
	}
	if xt == nil {
		return p.errorf("unrecognized extension %q", name)
	}
	fd := xt.TypeDescriptor()
	if fd.ContainingMessage().FullName() != m.Descriptor().FullName() {
		return p.errorf("extension field %q does not extend message %q", name, m.Descriptor().FullName())
	}

	if err := p.checkForColon(fd); err != nil {
		return err
	}

	v := m.Get(fd)
	if !m.Has(fd) && (fd.IsList() || fd.IsMap() || fd.Message() != nil) {
		v = m.Mutable(fd)
	}
	v, err = p.unmarshalValue(v, fd)
	if err != nil {
		return err
	}
	m.Set(fd, v)
	return p.consumeOptionalSeparator()
}

func (p *textParser) unmarshalValue(v protoreflect.Value, fd protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	tok := p.next()
	if tok.err != nil {
		return v, tok.err
	}
	if tok.value == "" {
		return v, p.errorf("unexpected EOF")
	}

	switch {
	case fd.IsList():
		lv := v.List()
		var err error
		if tok.value == "[" {
			// Repeated field with list notation, like [1,2,3].
			for {
				vv := lv.NewElement()
				vv, err = p.unmarshalSingularValue(vv, fd)
				if err != nil {
					return v, err
				}
				lv.Append(vv)

				tok := p.next()
				if tok.err != nil {
					return v, tok.err
				}
				if tok.value == "]" {
					break
				}
				if tok.value != "," {
					return v, p.errorf("Expected ']' or ',' found %q", tok.value)
				}
			}
			return v, nil
		}

		// One value of the repeated field.
		p.back()
		vv := lv.NewElement()
		vv, err = p.unmarshalSingularValue(vv, fd)
		if err != nil {
			return v, err
		}
		lv.Append(vv)
		return v, nil
	case fd.IsMap():
		// The map entry should be this sequence of tokens:
		//	< key : KEY value : VALUE >
		// However, implementations may omit key or value, and technically
		// we should support them in any order.
		var terminator string
		switch tok.value {
		case "<":
			terminator = ">"
		case "{":
			terminator = "}"
		default:
			return v, p.errorf("expected '{' or '<', found %q", tok.value)
		}

		keyFD := fd.MapKey()
		valFD := fd.MapValue()

		mv := v.Map()
		kv := keyFD.Default()
		vv := mv.NewValue()
		for {
			tok := p.next()
			if tok.err != nil {
				return v, tok.err
			}
			if tok.value == terminator {
				break
			}
			var err error
			switch tok.value {
			case "key":
				if err := p.consumeToken(":"); err != nil {
					return v, err
				}
				if kv, err = p.unmarshalSingularValue(kv, keyFD); err != nil {
					return v, err
				}
				if err := p.consumeOptionalSeparator(); err != nil {
					return v, err
				}
			case "value":
				if err := p.checkForColon(valFD); err != nil {
					return v, err
				}
				if vv, err = p.unmarshalSingularValue(vv, valFD); err != nil {
					return v, err
				}
				if err := p.consumeOptionalSeparator(); err != nil {
					return v, err
				}
			default:
				p.back()
				return v, p.errorf(`expected "key", "value", or %q, found %q`, terminator, tok.value)
			}
		}
		mv.Set(kv.MapKey(), vv)
		return v, nil
	default:
		p.back()
		return p.unmarshalSingularValue(v, fd)
	}
}

func (p *textParser) unmarshalSingularValue(v protoreflect.Value, fd protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	tok := p.next()
	if tok.err != nil {
		return v, tok.err
	}
	if tok.value == "" {
		return v, p.errorf("unexpected EOF")
	}

	switch fd.Kind() {
	case protoreflect.BoolKind:
		switch tok.value {
		case "true", "1", "t", "True":
			return protoreflect.ValueOfBool(true), nil
		case "false", "0", "f", "False":
			return protoreflect.ValueOfBool(false), nil
		}
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		if x, err := strconv.ParseInt(tok.value, 0, 32); err == nil {
			return protoreflect.ValueOfInt32(int32(x)), nil
		}

		// The C++ parser accepts large positive hex numbers that uses
		// two's complement arithmetic to represent negative numbers.
		// This feature is here for backwards compatibility with C++.
		if strings.HasPrefix(tok.value, "0x") {
			if x, err := strconv.ParseUint(tok.value, 0, 32); err == nil {
				return protoreflect.ValueOfInt32(int32(-(int64(^x) + 1))), nil
			}
		}
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		if x, err := strconv.ParseInt(tok.value, 0, 64); err == nil {
			return protoreflect.ValueOfInt64(int64(x)), nil
		}

		// The C++ parser accepts large positive hex numbers that uses
		// two's complement arithmetic to represent negative numbers.
		// This feature is here for backwards compatibility with C++.
		if strings.HasPrefix(tok.value, "0x") {
			if x, err := strconv.ParseUint(tok.value, 0, 64); err == nil {
				return protoreflect.ValueOfInt64(int64(-(int64(^x) + 1))), nil
			}
		}
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		if x, err := strconv.ParseUint(tok.value, 0, 32); err == nil {
			return protoreflect.ValueOfUint32(uint32(x)), nil
		}
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		if x, err := strconv.ParseUint(tok.value, 0, 64); err == nil {
			return protoreflect.ValueOfUint64(uint64(x)), nil
		}
	case protoreflect.FloatKind:
		// Ignore 'f' for compatibility with output generated by C++,
		// but don't remove 'f' when the value is "-inf" or "inf".
		v := tok.value
		if strings.HasSuffix(v, "f") && v != "-inf" && v != "inf" {
			v = v[:len(v)-len("f")]
		}
		if x, err := strconv.ParseFloat(v, 32); err == nil {
			return protoreflect.ValueOfFloat32(float32(x)), nil
		}
	case protoreflect.DoubleKind:
		// Ignore 'f' for compatibility with output generated by C++,
		// but don't remove 'f' when the value is "-inf" or "inf".
		v := tok.value
		if strings.HasSuffix(v, "f") && v != "-inf" && v != "inf" {
			v = v[:len(v)-len("f")]
		}
		if x, err := strconv.ParseFloat(v, 64); err == nil {
			return protoreflect.ValueOfFloat64(float64(x)), nil
		}
	case protoreflect.StringKind:
		if isQuote(tok.value[0]) {
			return protoreflect.ValueOfString(tok.unquoted), nil
		}
	case protoreflect.BytesKind:
		if isQuote(tok.value[0]) {
			return protoreflect.ValueOfBytes([]byte(tok.unquoted)), nil
		}
	case protoreflect.EnumKind:
		if x, err := strconv.ParseInt(tok.value, 0, 32); err == nil {
			return protoreflect.ValueOfEnum(protoreflect.EnumNumber(x)), nil
		}
		vd := fd.Enum().Values().ByName(protoreflect.Name(tok.value))
		if vd != nil {
			return protoreflect.ValueOfEnum(vd.Number()), nil
		}
	case protoreflect.MessageKind, protoreflect.GroupKind:
		var terminator string
		switch tok.value {
		case "{":
			terminator = "}"
		case "<":
			terminator = ">"
		default:
			return v, p.errorf("expected '{' or '<', found %q", tok.value)
		}
		err := p.unmarshalMessage(v.Message(), terminator)
		return v, err
	default:
		panic(fmt.Sprintf("invalid kind %v", fd.Kind()))
	}
	return v, p.errorf("invalid %v: %v", fd.Kind(), tok.value)
}

// Consume a ':' from the input stream (if the next token is a colon),
// returning an error if a colon is needed but not present.
func (p *textParser) checkForColon(fd protoreflect.FieldDescriptor) *ParseError {
	tok := p.next()
	if tok.err != nil {
		return tok.err
	}
	if tok.value != ":" {
		if fd.Message() == nil {
			return p.errorf("expected ':', found %q", tok.value)
		}
		p.back()
	}
	return nil
}

// consumeExtensionOrAnyName consumes an extension name or an Any type URL and
// the following ']'. It returns the name or URL consumed.
func (p *textParser) consumeExtensionOrAnyName() (string, error) {
	tok := p.next()
	if tok.err != nil {
		return "", tok.err
	}

	// If extension name or type url is quoted, it's a single token.
	if len(tok.value) > 2 && isQuote(tok.value[0]) && tok.value[len(tok.value)-1] == tok.value[0] {
		name, err := unquoteC(tok.value[1:len(tok.value)-1], rune(tok.value[0]))
		if err != nil {
			return "", err
		}
		return name, p.consumeToken("]")
	}

	// Consume everything up to "]"
	var parts []string
	for tok.value != "]" {
		parts = append(parts, tok.value)
		tok = p.next()
		if tok.err != nil {
			return "", p.errorf("unrecognized type_url or extension name: %s", tok.err)
		}
		if p.done && tok.value != "]" {
			return "", p.errorf("unclosed type_url or extension name")
		}
	}
	return strings.Join(parts, ""), nil
}

// consumeOptionalSeparator consumes an optional semicolon or comma.
// It is used in unmarshalMessage to provide backward compatibility.
func (p *textParser) consumeOptionalSeparator() error {
	tok := p.next()
	if tok.err != nil {
		return tok.err
	}
	if tok.value != ";" && tok.value != "," {
		p.back()
	}
	return nil
}

func (p *textParser) errorf(format string, a ...interface{}) *ParseError {
	pe := &ParseError{fmt.Sprintf(format, a...), p.cur.line, p.cur.offset}
	p.cur.err = pe
	p.done = true
	return pe
}

func (p *textParser) skipWhitespace() {
	i := 0
	for i < len(p.s) && (isWhitespace(p.s[i]) || p.s[i] == '#') {
		if p.s[i] == '#' {
			// comment; skip to end of line or input
			for i < len(p.s) && p.s[i] != '\n' {
				i++
			}
			if i == len(p.s) {
				break
			}
		}
		if p.s[i] == '\n' {
			p.line++
		}
		i++
	}
	p.offset += i
	p.s = p.s[i:len(p.s)]
	if len(p.s) == 0 {
		p.done = true
	}
}

func (p *textParser) advance() {
	// Skip whitespace
	p.skipWhitespace()
	if p.done {
		return
	}

	// Start of non-whitespace
	p.cur.err = nil
	p.cur.offset, p.cur.line = p.offset, p.line
	p.cur.unquoted = ""
	switch p.s[0] {
	case '<', '>', '{', '}', ':', '[', ']', ';', ',', '/':
		// Single symbol
		p.cur.value, p.s = p.s[0:1], p.s[1:len(p.s)]
	case '"', '\'':
		// Quoted string
		i := 1
		for i < len(p.s) && p.s[i] != p.s[0] && p.s[i] != '\n' {
			if p.s[i] == '\\' && i+1 < len(p.s) {
				// skip escaped char
				i++
			}
			i++
		}
		if i >= len(p.s) || p.s[i] != p.s[0] {
			p.errorf("unmatched quote")
			return
		}
		unq, err := unquoteC(p.s[1:i], rune(p.s[0]))
		if err != nil {
			p.errorf("invalid quoted string %s: %v", p.s[0:i+1], err)
			return
		}
		p.cur.value, p.s = p.s[0:i+1], p.s[i+1:len(p.s)]
		p.cur.unquoted = unq
	default:
		i := 0
		for i < len(p.s) && isIdentOrNumberChar(p.s[i]) {
			i++
		}
		if i == 0 {
			p.errorf("unexpected byte %#x", p.s[0])
			return
		}
		p.cur.value, p.s = p.s[0:i], p.s[i:len(p.s)]
	}
	p.offset += len(p.cur.value)
}

// Back off the parser by one token. Can only be done between calls to next().
// It makes the next advance() a no-op.
func (p *textParser) back() { p.backed = true }

// Advances the parser and returns the new current token.
func (p *textParser) next() *token {
	if p.backed || p.done {
		p.backed = false
		return &p.cur
	}
	p.advance()
	if p.done {
		p.cur.value = ""
	} else if len(p.cur.value) > 0 && isQuote(p.cur.value[0]) {
		// Look for multiple quoted strings separated by whitespace,
		// and concatenate them.
		cat := p.cur
		for {
			p.skipWhitespace()
			if p.done || !isQuote(p.s[0]) {
				break
			}
			p.advance()
			if p.cur.err != nil {
				return &p.cur
			}
			cat.value += " " + p.cur.value
			cat.unquoted += p.cur.unquoted
		}
		p.done = false // parser may have seen EOF, but we want to return cat
		p.cur = cat
	}
	return &p.cur
}

func (p *textParser) consumeToken(s string) error {
	tok := p.next()
	if tok.err != nil {
		return tok.err
	}
	if tok.value != s {
		p.back()
		return p.errorf("expected %q, found %q", s, tok.value)
	}
	return nil
}

var errBadUTF8 = errors.New("proto: bad UTF-8")

func unquoteC(s string, quote rune) (string, error) {
	// This is based on C++'s tokenizer.cc.
	// Despite its name, this is *not* parsing C syntax.
	// For instance, "\0" is an invalid quoted string.

	// Avoid allocation in trivial cases.
	simple := true
	for _, r := range s {
		if r == '\\' || r == quote {
			simple = false
			break
		}
	}
	if simple {
		return s, nil
	}

	buf := make([]byte, 0, 3*len(s)/2)
	for len(s) > 0 {
		r, n := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && n == 1 {
			return "", errBadUTF8
		}
		s = s[n:]
		if r != '\\' {
			if r < utf8.RuneSelf {
				buf = append(buf, byte(r))
			} else {
				buf = append(buf, string(r)...)
			}
			continue
		}

		ch, tail, err := unescape(s)
		if err != nil {
			return "", err
		}
		buf = append(buf, ch...)
		s = tail
	}
	return string(buf), nil
}

func unescape(s string) (ch string, tail string, err error) {
	r, n := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError && n == 1 {
		return "", "", errBadUTF8
	}
	s = s[n:]
	switch r {
	case 'a':
		return "\a", s, nil
	case 'b':
		return "\b", s, nil
	case 'f':
		return "\f", s, nil
	case 'n':
		return "\n", s, nil
	case 'r':
		return "\r", s, nil
	case 't':
		return "\t", s, nil
	case 'v':
		return "\v", s, nil
	case '?':
		return "?", s, nil // trigraph workaround
	case '\'', '"', '\\':
		return string(r), s, nil
	case '0', '1', '2', '3', '4', '5', '6', '7':
		if len(s) < 2 {
			return "", "", fmt.Errorf(`\%c requires 2 following digits`, r)
		}
		ss := string(r) + s[:2]
		s = s[2:]
		i, err := strconv.ParseUint(ss, 8, 8)
		if err != nil {
			return "", "", fmt.Errorf(`\%s contains non-octal digits`, ss)
		}
		return string([]byte{byte(i)}), s, nil
	case 'x', 'X', 'u', 'U':
		var n int
		switch r {
		case 'x', 'X':
			n = 2
		case 'u':
			n = 4
		case 'U':
			n = 8
		}
		if len(s) < n {
			return "", "", fmt.Errorf(`\%c requires %d following digits`, r, n)
		}
		ss := s[:n]
		s = s[n:]
		i, err := strconv.ParseUint(ss, 16, 64)
		if err != nil {
			return "", "", fmt.Errorf(`\%c%s contains non-hexadecimal digits`, r, ss)
		}
		if r == 'x' || r == 'X' {
			return string([]byte{byte(i)}), s, nil
		}
		if i > utf8.MaxRune {
			return "", "", fmt.Errorf(`\%c%s is not a valid Unicode code point`, r, ss)
		}
		return string(rune(i)), s, nil
	}
	return "", "", fmt.Errorf(`unknown escape \%c`, r)
}

func isIdentOrNumberChar(c byte) bool {
	switch {
	case 'A' <= c && c <= 'Z', 'a' <= c && c <= 'z':
		return true
	case '0' <= c && c <= '9':
		return true
	}
	switch c {
	case '-', '+', '.', '_':
		return true
	}
	return false
}

func isWhitespace(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return false
}

func isQuote(c byte) bool {
	switch c {
	case '"', '\'':
		return true
	}
	return false
}
