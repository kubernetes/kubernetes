// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

// Functions for parsing the Text protocol buffer format.
// TODO: message sets.

import (
	"encoding"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Error string emitted when deserializing Any and fields are already set
const anyRepeatedlyUnpacked = "Any message unpacked multiple times, or %q already set"

type ParseError struct {
	Message string
	Line    int // 1-based line number
	Offset  int // 0-based byte offset from start of input
}

func (p *ParseError) Error() string {
	if p.Line == 1 {
		// show offset only for first line
		return fmt.Sprintf("line 1.%d: %v", p.Offset, p.Message)
	}
	return fmt.Sprintf("line %d: %v", p.Line, p.Message)
}

type token struct {
	value    string
	err      *ParseError
	line     int    // line number
	offset   int    // byte number from start of input, not start of line
	unquoted string // the unquoted version of value, if it was a quoted string
}

func (t *token) String() string {
	if t.err == nil {
		return fmt.Sprintf("%q (line=%d, offset=%d)", t.value, t.line, t.offset)
	}
	return fmt.Sprintf("parse error: %v", t.err)
}

type textParser struct {
	s            string // remaining input
	done         bool   // whether the parsing is finished (success or error)
	backed       bool   // whether back() was called
	offset, line int
	cur          token
}

func newTextParser(s string) *textParser {
	p := new(textParser)
	p.s = s
	p.line = 1
	p.cur.line = 1
	return p
}

func (p *textParser) errorf(format string, a ...interface{}) *ParseError {
	pe := &ParseError{fmt.Sprintf(format, a...), p.cur.line, p.cur.offset}
	p.cur.err = pe
	p.done = true
	return pe
}

// Numbers and identifiers are matched by [-+._A-Za-z0-9]
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

var (
	errBadUTF8 = errors.New("proto: bad UTF-8")
	errBadHex  = errors.New("proto: bad hexadecimal")
)

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
	case '0', '1', '2', '3', '4', '5', '6', '7', 'x', 'X':
		if len(s) < 2 {
			return "", "", fmt.Errorf(`\%c requires 2 following digits`, r)
		}
		base := 8
		ss := s[:2]
		s = s[2:]
		if r == 'x' || r == 'X' {
			base = 16
		} else {
			ss = string(r) + ss
		}
		i, err := strconv.ParseUint(ss, base, 8)
		if err != nil {
			return "", "", err
		}
		return string([]byte{byte(i)}), s, nil
	case 'u', 'U':
		n := 4
		if r == 'U' {
			n = 8
		}
		if len(s) < n {
			return "", "", fmt.Errorf(`\%c requires %d digits`, r, n)
		}

		bs := make([]byte, n/2)
		for i := 0; i < n; i += 2 {
			a, ok1 := unhex(s[i])
			b, ok2 := unhex(s[i+1])
			if !ok1 || !ok2 {
				return "", "", errBadHex
			}
			bs[i/2] = a<<4 | b
		}
		s = s[n:]
		return string(bs), s, nil
	}
	return "", "", fmt.Errorf(`unknown escape \%c`, r)
}

// Adapted from src/pkg/strconv/quote.go.
func unhex(b byte) (v byte, ok bool) {
	switch {
	case '0' <= b && b <= '9':
		return b - '0', true
	case 'a' <= b && b <= 'f':
		return b - 'a' + 10, true
	case 'A' <= b && b <= 'F':
		return b - 'A' + 10, true
	}
	return 0, false
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

// Return a RequiredNotSetError indicating which required field was not set.
func (p *textParser) missingRequiredFieldError(sv reflect.Value) *RequiredNotSetError {
	st := sv.Type()
	sprops := GetProperties(st)
	for i := 0; i < st.NumField(); i++ {
		if !isNil(sv.Field(i)) {
			continue
		}

		props := sprops.Prop[i]
		if props.Required {
			return &RequiredNotSetError{fmt.Sprintf("%v.%v", st, props.OrigName)}
		}
	}
	return &RequiredNotSetError{fmt.Sprintf("%v.<unknown field name>", st)} // should not happen
}

// Returns the index in the struct for the named field, as well as the parsed tag properties.
func structFieldByName(sprops *StructProperties, name string) (int, *Properties, bool) {
	i, ok := sprops.decoderOrigNames[name]
	if ok {
		return i, sprops.Prop[i], true
	}
	return -1, nil, false
}

// Consume a ':' from the input stream (if the next token is a colon),
// returning an error if a colon is needed but not present.
func (p *textParser) checkForColon(props *Properties, typ reflect.Type) *ParseError {
	tok := p.next()
	if tok.err != nil {
		return tok.err
	}
	if tok.value != ":" {
		// Colon is optional when the field is a group or message.
		needColon := true
		switch props.Wire {
		case "group":
			needColon = false
		case "bytes":
			// A "bytes" field is either a message, a string, or a repeated field;
			// those three become *T, *string and []T respectively, so we can check for
			// this field being a pointer to a non-string.
			if typ.Kind() == reflect.Ptr {
				// *T or *string
				if typ.Elem().Kind() == reflect.String {
					break
				}
			} else if typ.Kind() == reflect.Slice {
				// []T or []*T
				if typ.Elem().Kind() != reflect.Ptr {
					break
				}
			} else if typ.Kind() == reflect.String {
				// The proto3 exception is for a string field,
				// which requires a colon.
				break
			}
			needColon = false
		}
		if needColon {
			return p.errorf("expected ':', found %q", tok.value)
		}
		p.back()
	}
	return nil
}

func (p *textParser) readStruct(sv reflect.Value, terminator string) error {
	st := sv.Type()
	sprops := GetProperties(st)
	reqCount := sprops.reqCount
	var reqFieldErr error
	fieldSet := make(map[string]bool)
	// A struct is a sequence of "name: value", terminated by one of
	// '>' or '}', or the end of the input.  A name may also be
	// "[extension]" or "[type/url]".
	//
	// The whole struct can also be an expanded Any message, like:
	// [type/url] < ... struct contents ... >
	for {
		tok := p.next()
		if tok.err != nil {
			return tok.err
		}
		if tok.value == terminator {
			break
		}
		if tok.value == "[" {
			// Looks like an extension or an Any.
			//
			// TODO: Check whether we need to handle
			// namespace rooted names (e.g. ".something.Foo").
			extName, err := p.consumeExtName()
			if err != nil {
				return err
			}

			if s := strings.LastIndex(extName, "/"); s >= 0 {
				// If it contains a slash, it's an Any type URL.
				messageName := extName[s+1:]
				mt := MessageType(messageName)
				if mt == nil {
					return p.errorf("unrecognized message %q in google.protobuf.Any", messageName)
				}
				tok = p.next()
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
				v := reflect.New(mt.Elem())
				if pe := p.readStruct(v.Elem(), terminator); pe != nil {
					return pe
				}
				b, err := Marshal(v.Interface().(Message))
				if err != nil {
					return p.errorf("failed to marshal message of type %q: %v", messageName, err)
				}
				if fieldSet["type_url"] {
					return p.errorf(anyRepeatedlyUnpacked, "type_url")
				}
				if fieldSet["value"] {
					return p.errorf(anyRepeatedlyUnpacked, "value")
				}
				sv.FieldByName("TypeUrl").SetString(extName)
				sv.FieldByName("Value").SetBytes(b)
				fieldSet["type_url"] = true
				fieldSet["value"] = true
				continue
			}

			var desc *ExtensionDesc
			// This could be faster, but it's functional.
			// TODO: Do something smarter than a linear scan.
			for _, d := range RegisteredExtensions(reflect.New(st).Interface().(Message)) {
				if d.Name == extName {
					desc = d
					break
				}
			}
			if desc == nil {
				return p.errorf("unrecognized extension %q", extName)
			}

			props := &Properties{}
			props.Parse(desc.Tag)

			typ := reflect.TypeOf(desc.ExtensionType)
			if err := p.checkForColon(props, typ); err != nil {
				return err
			}

			rep := desc.repeated()

			// Read the extension structure, and set it in
			// the value we're constructing.
			var ext reflect.Value
			if !rep {
				ext = reflect.New(typ).Elem()
			} else {
				ext = reflect.New(typ.Elem()).Elem()
			}
			if err := p.readAny(ext, props); err != nil {
				if _, ok := err.(*RequiredNotSetError); !ok {
					return err
				}
				reqFieldErr = err
			}
			ep := sv.Addr().Interface().(Message)
			if !rep {
				SetExtension(ep, desc, ext.Interface())
			} else {
				old, err := GetExtension(ep, desc)
				var sl reflect.Value
				if err == nil {
					sl = reflect.ValueOf(old) // existing slice
				} else {
					sl = reflect.MakeSlice(typ, 0, 1)
				}
				sl = reflect.Append(sl, ext)
				SetExtension(ep, desc, sl.Interface())
			}
			if err := p.consumeOptionalSeparator(); err != nil {
				return err
			}
			continue
		}

		// This is a normal, non-extension field.
		name := tok.value
		var dst reflect.Value
		fi, props, ok := structFieldByName(sprops, name)
		if ok {
			dst = sv.Field(fi)
		} else if oop, ok := sprops.OneofTypes[name]; ok {
			// It is a oneof.
			props = oop.Prop
			nv := reflect.New(oop.Type.Elem())
			dst = nv.Elem().Field(0)
			field := sv.Field(oop.Field)
			if !field.IsNil() {
				return p.errorf("field '%s' would overwrite already parsed oneof '%s'", name, sv.Type().Field(oop.Field).Name)
			}
			field.Set(nv)
		}
		if !dst.IsValid() {
			return p.errorf("unknown field name %q in %v", name, st)
		}

		if dst.Kind() == reflect.Map {
			// Consume any colon.
			if err := p.checkForColon(props, dst.Type()); err != nil {
				return err
			}

			// Construct the map if it doesn't already exist.
			if dst.IsNil() {
				dst.Set(reflect.MakeMap(dst.Type()))
			}
			key := reflect.New(dst.Type().Key()).Elem()
			val := reflect.New(dst.Type().Elem()).Elem()

			// The map entry should be this sequence of tokens:
			//	< key : KEY value : VALUE >
			// However, implementations may omit key or value, and technically
			// we should support them in any order.  See b/28924776 for a time
			// this went wrong.

			tok := p.next()
			var terminator string
			switch tok.value {
			case "<":
				terminator = ">"
			case "{":
				terminator = "}"
			default:
				return p.errorf("expected '{' or '<', found %q", tok.value)
			}
			for {
				tok := p.next()
				if tok.err != nil {
					return tok.err
				}
				if tok.value == terminator {
					break
				}
				switch tok.value {
				case "key":
					if err := p.consumeToken(":"); err != nil {
						return err
					}
					if err := p.readAny(key, props.mkeyprop); err != nil {
						return err
					}
					if err := p.consumeOptionalSeparator(); err != nil {
						return err
					}
				case "value":
					if err := p.checkForColon(props.mvalprop, dst.Type().Elem()); err != nil {
						return err
					}
					if err := p.readAny(val, props.mvalprop); err != nil {
						return err
					}
					if err := p.consumeOptionalSeparator(); err != nil {
						return err
					}
				default:
					p.back()
					return p.errorf(`expected "key", "value", or %q, found %q`, terminator, tok.value)
				}
			}

			dst.SetMapIndex(key, val)
			continue
		}

		// Check that it's not already set if it's not a repeated field.
		if !props.Repeated && fieldSet[name] {
			return p.errorf("non-repeated field %q was repeated", name)
		}

		if err := p.checkForColon(props, dst.Type()); err != nil {
			return err
		}

		// Parse into the field.
		fieldSet[name] = true
		if err := p.readAny(dst, props); err != nil {
			if _, ok := err.(*RequiredNotSetError); !ok {
				return err
			}
			reqFieldErr = err
		}
		if props.Required {
			reqCount--
		}

		if err := p.consumeOptionalSeparator(); err != nil {
			return err
		}

	}

	if reqCount > 0 {
		return p.missingRequiredFieldError(sv)
	}
	return reqFieldErr
}

// consumeExtName consumes extension name or expanded Any type URL and the
// following ']'. It returns the name or URL consumed.
func (p *textParser) consumeExtName() (string, error) {
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
	}
	return strings.Join(parts, ""), nil
}

// consumeOptionalSeparator consumes an optional semicolon or comma.
// It is used in readStruct to provide backward compatibility.
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

func (p *textParser) readAny(v reflect.Value, props *Properties) error {
	tok := p.next()
	if tok.err != nil {
		return tok.err
	}
	if tok.value == "" {
		return p.errorf("unexpected EOF")
	}

	switch fv := v; fv.Kind() {
	case reflect.Slice:
		at := v.Type()
		if at.Elem().Kind() == reflect.Uint8 {
			// Special case for []byte
			if tok.value[0] != '"' && tok.value[0] != '\'' {
				// Deliberately written out here, as the error after
				// this switch statement would write "invalid []byte: ...",
				// which is not as user-friendly.
				return p.errorf("invalid string: %v", tok.value)
			}
			bytes := []byte(tok.unquoted)
			fv.Set(reflect.ValueOf(bytes))
			return nil
		}
		// Repeated field.
		if tok.value == "[" {
			// Repeated field with list notation, like [1,2,3].
			for {
				fv.Set(reflect.Append(fv, reflect.New(at.Elem()).Elem()))
				err := p.readAny(fv.Index(fv.Len()-1), props)
				if err != nil {
					return err
				}
				tok := p.next()
				if tok.err != nil {
					return tok.err
				}
				if tok.value == "]" {
					break
				}
				if tok.value != "," {
					return p.errorf("Expected ']' or ',' found %q", tok.value)
				}
			}
			return nil
		}
		// One value of the repeated field.
		p.back()
		fv.Set(reflect.Append(fv, reflect.New(at.Elem()).Elem()))
		return p.readAny(fv.Index(fv.Len()-1), props)
	case reflect.Bool:
		// true/1/t/True or false/f/0/False.
		switch tok.value {
		case "true", "1", "t", "True":
			fv.SetBool(true)
			return nil
		case "false", "0", "f", "False":
			fv.SetBool(false)
			return nil
		}
	case reflect.Float32, reflect.Float64:
		v := tok.value
		// Ignore 'f' for compatibility with output generated by C++, but don't
		// remove 'f' when the value is "-inf" or "inf".
		if strings.HasSuffix(v, "f") && tok.value != "-inf" && tok.value != "inf" {
			v = v[:len(v)-1]
		}
		if f, err := strconv.ParseFloat(v, fv.Type().Bits()); err == nil {
			fv.SetFloat(f)
			return nil
		}
	case reflect.Int32:
		if x, err := strconv.ParseInt(tok.value, 0, 32); err == nil {
			fv.SetInt(x)
			return nil
		}

		if len(props.Enum) == 0 {
			break
		}
		m, ok := enumValueMaps[props.Enum]
		if !ok {
			break
		}
		x, ok := m[tok.value]
		if !ok {
			break
		}
		fv.SetInt(int64(x))
		return nil
	case reflect.Int64:
		if x, err := strconv.ParseInt(tok.value, 0, 64); err == nil {
			fv.SetInt(x)
			return nil
		}

	case reflect.Ptr:
		// A basic field (indirected through pointer), or a repeated message/group
		p.back()
		fv.Set(reflect.New(fv.Type().Elem()))
		return p.readAny(fv.Elem(), props)
	case reflect.String:
		if tok.value[0] == '"' || tok.value[0] == '\'' {
			fv.SetString(tok.unquoted)
			return nil
		}
	case reflect.Struct:
		var terminator string
		switch tok.value {
		case "{":
			terminator = "}"
		case "<":
			terminator = ">"
		default:
			return p.errorf("expected '{' or '<', found %q", tok.value)
		}
		// TODO: Handle nested messages which implement encoding.TextUnmarshaler.
		return p.readStruct(fv, terminator)
	case reflect.Uint32:
		if x, err := strconv.ParseUint(tok.value, 0, 32); err == nil {
			fv.SetUint(x)
			return nil
		}
	case reflect.Uint64:
		if x, err := strconv.ParseUint(tok.value, 0, 64); err == nil {
			fv.SetUint(x)
			return nil
		}
	}
	return p.errorf("invalid %v: %v", v.Type(), tok.value)
}

// UnmarshalText reads a protocol buffer in Text format. UnmarshalText resets pb
// before starting to unmarshal, so any existing data in pb is always removed.
// If a required field is not set and no other error occurs,
// UnmarshalText returns *RequiredNotSetError.
func UnmarshalText(s string, pb Message) error {
	if um, ok := pb.(encoding.TextUnmarshaler); ok {
		err := um.UnmarshalText([]byte(s))
		return err
	}
	pb.Reset()
	v := reflect.ValueOf(pb)
	if pe := newTextParser(s).readStruct(v.Elem(), ""); pe != nil {
		return pe
	}
	return nil
}
