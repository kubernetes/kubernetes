// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"io"
	"math"
	"math/bits"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

// EncodeOptions configures how JSON encoding operates.
// The zero value is equivalent to the default settings,
// which is compliant with both RFC 7493 and RFC 8259.
type EncodeOptions struct {
	requireKeyedLiterals
	nonComparable

	// multiline specifies whether the encoder should emit multiline output.
	multiline bool

	// omitTopLevelNewline specifies whether to omit the newline
	// that is appended after every top-level JSON value when streaming.
	omitTopLevelNewline bool

	// AllowDuplicateNames specifies that JSON objects may contain
	// duplicate member names. Disabling the duplicate name check may provide
	// performance benefits, but breaks compliance with RFC 7493, section 2.3.
	// The output will still be compliant with RFC 8259,
	// which leaves the handling of duplicate names as unspecified behavior.
	AllowDuplicateNames bool

	// AllowInvalidUTF8 specifies that JSON strings may contain invalid UTF-8,
	// which will be mangled as the Unicode replacement character, U+FFFD.
	// This causes the encoder to break compliance with
	// RFC 7493, section 2.1, and RFC 8259, section 8.1.
	AllowInvalidUTF8 bool

	// preserveRawStrings specifies that WriteToken and WriteValue should not
	// reformat any JSON string, but keep the formatting verbatim.
	preserveRawStrings bool

	// canonicalizeNumbers specifies that WriteToken and WriteValue should
	// reformat any JSON numbers according to RFC 8785, section 3.2.2.3.
	canonicalizeNumbers bool

	// EscapeRune reports whether the provided character should be escaped
	// as a hexadecimal Unicode codepoint (e.g., \ufffd).
	// If nil, the shortest and simplest encoding will be used,
	// which is also the formatting specified by RFC 8785, section 3.2.2.2.
	EscapeRune func(rune) bool

	// Indent (if non-empty) specifies that the encoder should emit multiline
	// output where each element in a JSON object or array begins on a new,
	// indented line beginning with the indent prefix followed by one or more
	// copies of indent according to the indentation nesting.
	// It may only be composed of space or tab characters.
	Indent string

	// IndentPrefix is prepended to each line within a JSON object or array.
	// The purpose of the indent prefix is to encode data that can more easily
	// be embedded inside other formatted JSON data.
	// It may only be composed of space or tab characters.
	// It is ignored if Indent is empty.
	IndentPrefix string
}

// Encoder is a streaming encoder from raw JSON tokens and values.
// It is used to write a stream of top-level JSON values,
// each terminated with a newline character.
//
// WriteToken and WriteValue calls may be interleaved.
// For example, the following JSON value:
//
//	{"name":"value","array":[null,false,true,3.14159],"object":{"k":"v"}}
//
// can be composed with the following calls (ignoring errors for brevity):
//
//	e.WriteToken(ObjectStart)           // {
//	e.WriteToken(String("name"))        // "name"
//	e.WriteToken(String("value"))       // "value"
//	e.WriteValue(RawValue(`"array"`))   // "array"
//	e.WriteToken(ArrayStart)            // [
//	e.WriteToken(Null)                  // null
//	e.WriteToken(False)                 // false
//	e.WriteValue(RawValue("true"))      // true
//	e.WriteToken(Float(3.14159))        // 3.14159
//	e.WriteToken(ArrayEnd)              // ]
//	e.WriteValue(RawValue(`"object"`))  // "object"
//	e.WriteValue(RawValue(`{"k":"v"}`)) // {"k":"v"}
//	e.WriteToken(ObjectEnd)             // }
//
// The above is one of many possible sequence of calls and
// may not represent the most sensible method to call for any given token/value.
// For example, it is probably more common to call WriteToken with a string
// for object names.
type Encoder struct {
	state
	encodeBuffer
	options EncodeOptions

	seenPointers seenPointers // only used when marshaling
}

// encodeBuffer is a buffer split into 2 segments:
//
//   - buf[0:len(buf)]        // written (but unflushed) portion of the buffer
//   - buf[len(buf):cap(buf)] // unused portion of the buffer
type encodeBuffer struct {
	buf []byte // may alias wr if it is a bytes.Buffer

	// baseOffset is added to len(buf) to obtain the absolute offset
	// relative to the start of io.Writer stream.
	baseOffset int64

	wr io.Writer

	// maxValue is the approximate maximum RawValue size passed to WriteValue.
	maxValue int
	// unusedCache is the buffer returned by the UnusedBuffer method.
	unusedCache []byte
	// bufStats is statistics about buffer utilization.
	// It is only used with pooled encoders in pools.go.
	bufStats bufferStatistics
}

// NewEncoder constructs a new streaming encoder writing to w.
func NewEncoder(w io.Writer) *Encoder {
	return EncodeOptions{}.NewEncoder(w)
}

// NewEncoder constructs a new streaming encoder writing to w
// configured with the provided options.
// It flushes the internal buffer when the buffer is sufficiently full or
// when a top-level value has been written.
//
// If w is a bytes.Buffer, then the encoder appends directly into the buffer
// without copying the contents from an intermediate buffer.
func (o EncodeOptions) NewEncoder(w io.Writer) *Encoder {
	e := new(Encoder)
	o.ResetEncoder(e, w)
	return e
}

// ResetEncoder resets an encoder such that it is writing afresh to w and
// configured with the provided options.
func (o EncodeOptions) ResetEncoder(e *Encoder, w io.Writer) {
	if e == nil {
		panic("json: invalid nil Encoder")
	}
	if w == nil {
		panic("json: invalid nil io.Writer")
	}
	e.reset(nil, w, o)
}

func (e *Encoder) reset(b []byte, w io.Writer, o EncodeOptions) {
	if len(o.Indent) > 0 {
		o.multiline = true
		if s := trimLeftSpaceTab(o.IndentPrefix); len(s) > 0 {
			panic("json: invalid character " + quoteRune([]byte(s)) + " in indent prefix")
		}
		if s := trimLeftSpaceTab(o.Indent); len(s) > 0 {
			panic("json: invalid character " + quoteRune([]byte(s)) + " in indent")
		}
	}
	e.state.reset()
	e.encodeBuffer = encodeBuffer{buf: b, wr: w, bufStats: e.bufStats}
	e.options = o
	if bb, ok := w.(*bytes.Buffer); ok && bb != nil {
		e.buf = bb.Bytes()[bb.Len():] // alias the unused buffer of bb
	}
}

// Reset resets an encoder such that it is writing afresh to w but
// keeps any pre-existing encoder options.
func (e *Encoder) Reset(w io.Writer) {
	e.options.ResetEncoder(e, w)
}

// needFlush determines whether to flush at this point.
func (e *Encoder) needFlush() bool {
	// NOTE: This function is carefully written to be inlineable.

	// Avoid flushing if e.wr is nil since there is no underlying writer.
	// Flush if less than 25% of the capacity remains.
	// Flushing at some constant fraction ensures that the buffer stops growing
	// so long as the largest Token or Value fits within that unused capacity.
	return e.wr != nil && (e.tokens.depth() == 1 || len(e.buf) > 3*cap(e.buf)/4)
}

// flush flushes the buffer to the underlying io.Writer.
// It may append a trailing newline after the top-level value.
func (e *Encoder) flush() error {
	if e.wr == nil || e.avoidFlush() {
		return nil
	}

	// In streaming mode, always emit a newline after the top-level value.
	if e.tokens.depth() == 1 && !e.options.omitTopLevelNewline {
		e.buf = append(e.buf, '\n')
	}

	// Inform objectNameStack that we are about to flush the buffer content.
	e.names.copyQuotedBuffer(e.buf)

	// Specialize bytes.Buffer for better performance.
	if bb, ok := e.wr.(*bytes.Buffer); ok {
		// If e.buf already aliases the internal buffer of bb,
		// then the Write call simply increments the internal offset,
		// otherwise Write operates as expected.
		// See https://go.dev/issue/42986.
		n, _ := bb.Write(e.buf) // never fails unless bb is nil
		e.baseOffset += int64(n)

		// If the internal buffer of bytes.Buffer is too small,
		// append operations elsewhere in the Encoder may grow the buffer.
		// This would be semantically correct, but hurts performance.
		// As such, ensure 25% of the current length is always available
		// to reduce the probability that other appends must allocate.
		if avail := bb.Cap() - bb.Len(); avail < bb.Len()/4 {
			bb.Grow(avail + 1)
		}

		e.buf = bb.Bytes()[bb.Len():] // alias the unused buffer of bb
		return nil
	}

	// Flush the internal buffer to the underlying io.Writer.
	n, err := e.wr.Write(e.buf)
	e.baseOffset += int64(n)
	if err != nil {
		// In the event of an error, preserve the unflushed portion.
		// Thus, write errors aren't fatal so long as the io.Writer
		// maintains consistent state after errors.
		if n > 0 {
			e.buf = e.buf[:copy(e.buf, e.buf[n:])]
		}
		return &ioError{action: "write", err: err}
	}
	e.buf = e.buf[:0]

	// Check whether to grow the buffer.
	// Note that cap(e.buf) may already exceed maxBufferSize since
	// an append elsewhere already grew it to store a large token.
	const maxBufferSize = 4 << 10
	const growthSizeFactor = 2 // higher value is faster
	const growthRateFactor = 2 // higher value is slower
	// By default, grow if below the maximum buffer size.
	grow := cap(e.buf) <= maxBufferSize/growthSizeFactor
	// Growing can be expensive, so only grow
	// if a sufficient number of bytes have been processed.
	grow = grow && int64(cap(e.buf)) < e.previousOffsetEnd()/growthRateFactor
	if grow {
		e.buf = make([]byte, 0, cap(e.buf)*growthSizeFactor)
	}

	return nil
}

func (e *encodeBuffer) previousOffsetEnd() int64 { return e.baseOffset + int64(len(e.buf)) }
func (e *encodeBuffer) unflushedBuffer() []byte  { return e.buf }

// avoidFlush indicates whether to avoid flushing to ensure there is always
// enough in the buffer to unwrite the last object member if it were empty.
func (e *Encoder) avoidFlush() bool {
	switch {
	case e.tokens.last.length() == 0:
		// Never flush after ObjectStart or ArrayStart since we don't know yet
		// if the object or array will end up being empty.
		return true
	case e.tokens.last.needObjectValue():
		// Never flush before the object value since we don't know yet
		// if the object value will end up being empty.
		return true
	case e.tokens.last.needObjectName() && len(e.buf) >= 2:
		// Never flush after the object value if it does turn out to be empty.
		switch string(e.buf[len(e.buf)-2:]) {
		case `ll`, `""`, `{}`, `[]`: // last two bytes of every empty value
			return true
		}
	}
	return false
}

// unwriteEmptyObjectMember unwrites the last object member if it is empty
// and reports whether it performed an unwrite operation.
func (e *Encoder) unwriteEmptyObjectMember(prevName *string) bool {
	if last := e.tokens.last; !last.isObject() || !last.needObjectName() || last.length() == 0 {
		panic("BUG: must be called on an object after writing a value")
	}

	// The flushing logic is modified to never flush a trailing empty value.
	// The encoder never writes trailing whitespace eagerly.
	b := e.unflushedBuffer()

	// Detect whether the last value was empty.
	var n int
	if len(b) >= 3 {
		switch string(b[len(b)-2:]) {
		case "ll": // last two bytes of `null`
			n = len(`null`)
		case `""`:
			// It is possible for a non-empty string to have `""` as a suffix
			// if the second to the last quote was escaped.
			if b[len(b)-3] == '\\' {
				return false // e.g., `"\""` is not empty
			}
			n = len(`""`)
		case `{}`:
			n = len(`{}`)
		case `[]`:
			n = len(`[]`)
		}
	}
	if n == 0 {
		return false
	}

	// Unwrite the value, whitespace, colon, name, whitespace, and comma.
	b = b[:len(b)-n]
	b = trimSuffixWhitespace(b)
	b = trimSuffixByte(b, ':')
	b = trimSuffixString(b)
	b = trimSuffixWhitespace(b)
	b = trimSuffixByte(b, ',')
	e.buf = b // store back truncated unflushed buffer

	// Undo state changes.
	e.tokens.last.decrement() // for object member value
	e.tokens.last.decrement() // for object member name
	if !e.options.AllowDuplicateNames {
		if e.tokens.last.isActiveNamespace() {
			e.namespaces.last().removeLast()
		}
		e.names.clearLast()
		if prevName != nil {
			e.names.copyQuotedBuffer(e.buf) // required by objectNameStack.replaceLastUnquotedName
			e.names.replaceLastUnquotedName(*prevName)
		}
	}
	return true
}

// unwriteOnlyObjectMemberName unwrites the only object member name
// and returns the unquoted name.
func (e *Encoder) unwriteOnlyObjectMemberName() string {
	if last := e.tokens.last; !last.isObject() || last.length() != 1 {
		panic("BUG: must be called on an object after writing first name")
	}

	// Unwrite the name and whitespace.
	b := trimSuffixString(e.buf)
	isVerbatim := bytes.IndexByte(e.buf[len(b):], '\\') < 0
	name := string(unescapeStringMayCopy(e.buf[len(b):], isVerbatim))
	e.buf = trimSuffixWhitespace(b)

	// Undo state changes.
	e.tokens.last.decrement()
	if !e.options.AllowDuplicateNames {
		if e.tokens.last.isActiveNamespace() {
			e.namespaces.last().removeLast()
		}
		e.names.clearLast()
	}
	return name
}

func trimSuffixWhitespace(b []byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	n := len(b) - 1
	for n >= 0 && (b[n] == ' ' || b[n] == '\t' || b[n] == '\r' || b[n] == '\n') {
		n--
	}
	return b[:n+1]
}

func trimSuffixString(b []byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	if len(b) > 0 && b[len(b)-1] == '"' {
		b = b[:len(b)-1]
	}
	for len(b) >= 2 && !(b[len(b)-1] == '"' && b[len(b)-2] != '\\') {
		b = b[:len(b)-1] // trim all characters except an unescaped quote
	}
	if len(b) > 0 && b[len(b)-1] == '"' {
		b = b[:len(b)-1]
	}
	return b
}

func hasSuffixByte(b []byte, c byte) bool {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	return len(b) > 0 && b[len(b)-1] == c
}

func trimSuffixByte(b []byte, c byte) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	if len(b) > 0 && b[len(b)-1] == c {
		return b[:len(b)-1]
	}
	return b
}

// WriteToken writes the next token and advances the internal write offset.
//
// The provided token kind must be consistent with the JSON grammar.
// For example, it is an error to provide a number when the encoder
// is expecting an object name (which is always a string), or
// to provide an end object delimiter when the encoder is finishing an array.
// If the provided token is invalid, then it reports a SyntacticError and
// the internal state remains unchanged.
func (e *Encoder) WriteToken(t Token) error {
	k := t.Kind()
	b := e.buf // use local variable to avoid mutating e in case of error

	// Append any delimiters or optional whitespace.
	b = e.tokens.mayAppendDelim(b, k)
	if e.options.multiline {
		b = e.appendWhitespace(b, k)
	}

	// Append the token to the output and to the state machine.
	var err error
	switch k {
	case 'n':
		b = append(b, "null"...)
		err = e.tokens.appendLiteral()
	case 'f':
		b = append(b, "false"...)
		err = e.tokens.appendLiteral()
	case 't':
		b = append(b, "true"...)
		err = e.tokens.appendLiteral()
	case '"':
		n0 := len(b) // offset before calling t.appendString
		if b, err = t.appendString(b, !e.options.AllowInvalidUTF8, e.options.preserveRawStrings, e.options.EscapeRune); err != nil {
			break
		}
		if !e.options.AllowDuplicateNames && e.tokens.last.needObjectName() {
			if !e.tokens.last.isValidNamespace() {
				err = errInvalidNamespace
				break
			}
			if e.tokens.last.isActiveNamespace() && !e.namespaces.last().insertQuoted(b[n0:], false) {
				err = &SyntacticError{str: "duplicate name " + string(b[n0:]) + " in object"}
				break
			}
			e.names.replaceLastQuotedOffset(n0) // only replace if insertQuoted succeeds
		}
		err = e.tokens.appendString()
	case '0':
		if b, err = t.appendNumber(b, e.options.canonicalizeNumbers); err != nil {
			break
		}
		err = e.tokens.appendNumber()
	case '{':
		b = append(b, '{')
		if err = e.tokens.pushObject(); err != nil {
			break
		}
		if !e.options.AllowDuplicateNames {
			e.names.push()
			e.namespaces.push()
		}
	case '}':
		b = append(b, '}')
		if err = e.tokens.popObject(); err != nil {
			break
		}
		if !e.options.AllowDuplicateNames {
			e.names.pop()
			e.namespaces.pop()
		}
	case '[':
		b = append(b, '[')
		err = e.tokens.pushArray()
	case ']':
		b = append(b, ']')
		err = e.tokens.popArray()
	default:
		return &SyntacticError{str: "invalid json.Token"}
	}
	if err != nil {
		return err
	}

	// Finish off the buffer and store it back into e.
	e.buf = b
	if e.needFlush() {
		return e.flush()
	}
	return nil
}

const (
	rawIntNumber  = -1
	rawUintNumber = -2
)

// writeNumber is specialized version of WriteToken, but optimized for numbers.
// As a special-case, if bits is -1 or -2, it will treat v as
// the raw-encoded bits of an int64 or uint64, respectively.
// It is only called from arshal_default.go.
func (e *Encoder) writeNumber(v float64, bits int, quote bool) error {
	b := e.buf // use local variable to avoid mutating e in case of error

	// Append any delimiters or optional whitespace.
	b = e.tokens.mayAppendDelim(b, '0')
	if e.options.multiline {
		b = e.appendWhitespace(b, '0')
	}

	if quote {
		// Append the value to the output.
		n0 := len(b) // offset before appending the number
		b = append(b, '"')
		switch bits {
		case rawIntNumber:
			b = strconv.AppendInt(b, int64(math.Float64bits(v)), 10)
		case rawUintNumber:
			b = strconv.AppendUint(b, uint64(math.Float64bits(v)), 10)
		default:
			b = appendNumber(b, v, bits)
		}
		b = append(b, '"')

		// Escape the string if necessary.
		if e.options.EscapeRune != nil {
			b2 := append(e.unusedCache, b[n0+len(`"`):len(b)-len(`"`)]...)
			b, _ = appendString(b[:n0], string(b2), false, e.options.EscapeRune)
			e.unusedCache = b2[:0]
		}

		// Update the state machine.
		if !e.options.AllowDuplicateNames && e.tokens.last.needObjectName() {
			if !e.tokens.last.isValidNamespace() {
				return errInvalidNamespace
			}
			if e.tokens.last.isActiveNamespace() && !e.namespaces.last().insertQuoted(b[n0:], false) {
				return &SyntacticError{str: "duplicate name " + string(b[n0:]) + " in object"}
			}
			e.names.replaceLastQuotedOffset(n0) // only replace if insertQuoted succeeds
		}
		if err := e.tokens.appendString(); err != nil {
			return err
		}
	} else {
		switch bits {
		case rawIntNumber:
			b = strconv.AppendInt(b, int64(math.Float64bits(v)), 10)
		case rawUintNumber:
			b = strconv.AppendUint(b, uint64(math.Float64bits(v)), 10)
		default:
			b = appendNumber(b, v, bits)
		}
		if err := e.tokens.appendNumber(); err != nil {
			return err
		}
	}

	// Finish off the buffer and store it back into e.
	e.buf = b
	if e.needFlush() {
		return e.flush()
	}
	return nil
}

// WriteValue writes the next raw value and advances the internal write offset.
// The Encoder does not simply copy the provided value verbatim, but
// parses it to ensure that it is syntactically valid and reformats it
// according to how the Encoder is configured to format whitespace and strings.
//
// The provided value kind must be consistent with the JSON grammar
// (see examples on Encoder.WriteToken). If the provided value is invalid,
// then it reports a SyntacticError and the internal state remains unchanged.
func (e *Encoder) WriteValue(v RawValue) error {
	e.maxValue |= len(v) // bitwise OR is a fast approximation of max

	k := v.Kind()
	b := e.buf // use local variable to avoid mutating e in case of error

	// Append any delimiters or optional whitespace.
	b = e.tokens.mayAppendDelim(b, k)
	if e.options.multiline {
		b = e.appendWhitespace(b, k)
	}

	// Append the value the output.
	var err error
	v = v[consumeWhitespace(v):]
	n0 := len(b) // offset before calling e.reformatValue
	b, v, err = e.reformatValue(b, v, e.tokens.depth())
	if err != nil {
		return err
	}
	v = v[consumeWhitespace(v):]
	if len(v) > 0 {
		return newInvalidCharacterError(v[0:], "after top-level value")
	}

	// Append the kind to the state machine.
	switch k {
	case 'n', 'f', 't':
		err = e.tokens.appendLiteral()
	case '"':
		if !e.options.AllowDuplicateNames && e.tokens.last.needObjectName() {
			if !e.tokens.last.isValidNamespace() {
				err = errInvalidNamespace
				break
			}
			if e.tokens.last.isActiveNamespace() && !e.namespaces.last().insertQuoted(b[n0:], false) {
				err = &SyntacticError{str: "duplicate name " + string(b[n0:]) + " in object"}
				break
			}
			e.names.replaceLastQuotedOffset(n0) // only replace if insertQuoted succeeds
		}
		err = e.tokens.appendString()
	case '0':
		err = e.tokens.appendNumber()
	case '{':
		if err = e.tokens.pushObject(); err != nil {
			break
		}
		if err = e.tokens.popObject(); err != nil {
			panic("BUG: popObject should never fail immediately after pushObject: " + err.Error())
		}
	case '[':
		if err = e.tokens.pushArray(); err != nil {
			break
		}
		if err = e.tokens.popArray(); err != nil {
			panic("BUG: popArray should never fail immediately after pushArray: " + err.Error())
		}
	}
	if err != nil {
		return err
	}

	// Finish off the buffer and store it back into e.
	e.buf = b
	if e.needFlush() {
		return e.flush()
	}
	return nil
}

// appendWhitespace appends whitespace that immediately precedes the next token.
func (e *Encoder) appendWhitespace(b []byte, next Kind) []byte {
	if e.tokens.needDelim(next) == ':' {
		return append(b, ' ')
	} else {
		return e.appendIndent(b, e.tokens.needIndent(next))
	}
}

// appendIndent appends the appropriate number of indentation characters
// for the current nested level, n.
func (e *Encoder) appendIndent(b []byte, n int) []byte {
	if n == 0 {
		return b
	}
	b = append(b, '\n')
	b = append(b, e.options.IndentPrefix...)
	for ; n > 1; n-- {
		b = append(b, e.options.Indent...)
	}
	return b
}

// reformatValue parses a JSON value from the start of src and
// appends it to the end of dst, reformatting whitespace and strings as needed.
// It returns the updated versions of dst and src.
func (e *Encoder) reformatValue(dst []byte, src RawValue, depth int) ([]byte, RawValue, error) {
	// TODO: Should this update valueFlags as input?
	if len(src) == 0 {
		return dst, src, io.ErrUnexpectedEOF
	}
	var n int
	var err error
	switch k := Kind(src[0]).normalize(); k {
	case 'n':
		if n = consumeNull(src); n == 0 {
			n, err = consumeLiteral(src, "null")
		}
	case 'f':
		if n = consumeFalse(src); n == 0 {
			n, err = consumeLiteral(src, "false")
		}
	case 't':
		if n = consumeTrue(src); n == 0 {
			n, err = consumeLiteral(src, "true")
		}
	case '"':
		if n := consumeSimpleString(src); n > 0 && e.options.EscapeRune == nil {
			dst, src = append(dst, src[:n]...), src[n:] // copy simple strings verbatim
			return dst, src, nil
		}
		return reformatString(dst, src, !e.options.AllowInvalidUTF8, e.options.preserveRawStrings, e.options.EscapeRune)
	case '0':
		if n := consumeSimpleNumber(src); n > 0 && !e.options.canonicalizeNumbers {
			dst, src = append(dst, src[:n]...), src[n:] // copy simple numbers verbatim
			return dst, src, nil
		}
		return reformatNumber(dst, src, e.options.canonicalizeNumbers)
	case '{':
		return e.reformatObject(dst, src, depth)
	case '[':
		return e.reformatArray(dst, src, depth)
	default:
		return dst, src, newInvalidCharacterError(src, "at start of value")
	}
	if err != nil {
		return dst, src, err
	}
	dst, src = append(dst, src[:n]...), src[n:]
	return dst, src, nil
}

// reformatObject parses a JSON object from the start of src and
// appends it to the end of src, reformatting whitespace and strings as needed.
// It returns the updated versions of dst and src.
func (e *Encoder) reformatObject(dst []byte, src RawValue, depth int) ([]byte, RawValue, error) {
	// Append object start.
	if src[0] != '{' {
		panic("BUG: reformatObject must be called with a buffer that starts with '{'")
	}
	dst, src = append(dst, '{'), src[1:]

	// Append (possible) object end.
	src = src[consumeWhitespace(src):]
	if len(src) == 0 {
		return dst, src, io.ErrUnexpectedEOF
	}
	if src[0] == '}' {
		dst, src = append(dst, '}'), src[1:]
		return dst, src, nil
	}

	var err error
	var names *objectNamespace
	if !e.options.AllowDuplicateNames {
		e.namespaces.push()
		defer e.namespaces.pop()
		names = e.namespaces.last()
	}
	depth++
	for {
		// Append optional newline and indentation.
		if e.options.multiline {
			dst = e.appendIndent(dst, depth)
		}

		// Append object name.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		n0 := len(dst) // offset before calling reformatString
		n := consumeSimpleString(src)
		if n > 0 && e.options.EscapeRune == nil {
			dst, src = append(dst, src[:n]...), src[n:] // copy simple strings verbatim
		} else {
			dst, src, err = reformatString(dst, src, !e.options.AllowInvalidUTF8, e.options.preserveRawStrings, e.options.EscapeRune)
		}
		if err != nil {
			return dst, src, err
		}
		if !e.options.AllowDuplicateNames && !names.insertQuoted(dst[n0:], false) {
			return dst, src, &SyntacticError{str: "duplicate name " + string(dst[n0:]) + " in object"}
		}

		// Append colon.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		if src[0] != ':' {
			return dst, src, newInvalidCharacterError(src, "after object name (expecting ':')")
		}
		dst, src = append(dst, ':'), src[1:]
		if e.options.multiline {
			dst = append(dst, ' ')
		}

		// Append object value.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		dst, src, err = e.reformatValue(dst, src, depth)
		if err != nil {
			return dst, src, err
		}

		// Append comma or object end.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		switch src[0] {
		case ',':
			dst, src = append(dst, ','), src[1:]
			continue
		case '}':
			if e.options.multiline {
				dst = e.appendIndent(dst, depth-1)
			}
			dst, src = append(dst, '}'), src[1:]
			return dst, src, nil
		default:
			return dst, src, newInvalidCharacterError(src, "after object value (expecting ',' or '}')")
		}
	}
}

// reformatArray parses a JSON array from the start of src and
// appends it to the end of dst, reformatting whitespace and strings as needed.
// It returns the updated versions of dst and src.
func (e *Encoder) reformatArray(dst []byte, src RawValue, depth int) ([]byte, RawValue, error) {
	// Append array start.
	if src[0] != '[' {
		panic("BUG: reformatArray must be called with a buffer that starts with '['")
	}
	dst, src = append(dst, '['), src[1:]

	// Append (possible) array end.
	src = src[consumeWhitespace(src):]
	if len(src) == 0 {
		return dst, src, io.ErrUnexpectedEOF
	}
	if src[0] == ']' {
		dst, src = append(dst, ']'), src[1:]
		return dst, src, nil
	}

	var err error
	depth++
	for {
		// Append optional newline and indentation.
		if e.options.multiline {
			dst = e.appendIndent(dst, depth)
		}

		// Append array value.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		dst, src, err = e.reformatValue(dst, src, depth)
		if err != nil {
			return dst, src, err
		}

		// Append comma or array end.
		src = src[consumeWhitespace(src):]
		if len(src) == 0 {
			return dst, src, io.ErrUnexpectedEOF
		}
		switch src[0] {
		case ',':
			dst, src = append(dst, ','), src[1:]
			continue
		case ']':
			if e.options.multiline {
				dst = e.appendIndent(dst, depth-1)
			}
			dst, src = append(dst, ']'), src[1:]
			return dst, src, nil
		default:
			return dst, src, newInvalidCharacterError(src, "after array value (expecting ',' or ']')")
		}
	}
}

// OutputOffset returns the current output byte offset. It gives the location
// of the next byte immediately after the most recently written token or value.
// The number of bytes actually written to the underlying io.Writer may be less
// than this offset due to internal buffering effects.
func (e *Encoder) OutputOffset() int64 {
	return e.previousOffsetEnd()
}

// UnusedBuffer returns a zero-length buffer with a possible non-zero capacity.
// This buffer is intended to be used to populate a RawValue
// being passed to an immediately succeeding WriteValue call.
//
// Example usage:
//
//	b := d.UnusedBuffer()
//	b = append(b, '"')
//	b = appendString(b, v) // append the string formatting of v
//	b = append(b, '"')
//	... := d.WriteValue(b)
//
// It is the user's responsibility to ensure that the value is valid JSON.
func (e *Encoder) UnusedBuffer() []byte {
	// NOTE: We don't return e.buf[len(e.buf):cap(e.buf)] since WriteValue would
	// need to take special care to avoid mangling the data while reformatting.
	// WriteValue can't easily identify whether the input RawValue aliases e.buf
	// without using unsafe.Pointer. Thus, we just return a different buffer.
	// Should this ever alias e.buf, we need to consider how it operates with
	// the specialized performance optimization for bytes.Buffer.
	n := 1 << bits.Len(uint(e.maxValue|63)) // fast approximation for max length
	if cap(e.unusedCache) < n {
		e.unusedCache = make([]byte, 0, n)
	}
	return e.unusedCache
}

// StackDepth returns the depth of the state machine for written JSON data.
// Each level on the stack represents a nested JSON object or array.
// It is incremented whenever an ObjectStart or ArrayStart token is encountered
// and decremented whenever an ObjectEnd or ArrayEnd token is encountered.
// The depth is zero-indexed, where zero represents the top-level JSON value.
func (e *Encoder) StackDepth() int {
	// NOTE: Keep in sync with Decoder.StackDepth.
	return e.tokens.depth() - 1
}

// StackIndex returns information about the specified stack level.
// It must be a number between 0 and StackDepth, inclusive.
// For each level, it reports the kind:
//
//   - 0 for a level of zero,
//   - '{' for a level representing a JSON object, and
//   - '[' for a level representing a JSON array.
//
// It also reports the length of that JSON object or array.
// Each name and value in a JSON object is counted separately,
// so the effective number of members would be half the length.
// A complete JSON object must have an even length.
func (e *Encoder) StackIndex(i int) (Kind, int) {
	// NOTE: Keep in sync with Decoder.StackIndex.
	switch s := e.tokens.index(i); {
	case i > 0 && s.isObject():
		return '{', s.length()
	case i > 0 && s.isArray():
		return '[', s.length()
	default:
		return 0, s.length()
	}
}

// StackPointer returns a JSON Pointer (RFC 6901) to the most recently written value.
// Object names are only present if AllowDuplicateNames is false, otherwise
// object members are represented using their index within the object.
func (e *Encoder) StackPointer() string {
	e.names.copyQuotedBuffer(e.buf)
	return string(e.appendStackPointer(nil))
}

// appendString appends src to dst as a JSON string per RFC 7159, section 7.
//
// If validateUTF8 is specified, this rejects input that contains invalid UTF-8
// otherwise invalid bytes are replaced with the Unicode replacement character.
// If escapeRune is provided, it specifies which runes to escape using
// hexadecimal sequences. If nil, the shortest representable form is used,
// which is also the canonical form for strings (RFC 8785, section 3.2.2.2).
//
// Note that this API allows full control over the formatting of strings
// except for whether a forward solidus '/' may be formatted as '\/' and
// the casing of hexadecimal Unicode escape sequences.
func appendString(dst []byte, src string, validateUTF8 bool, escapeRune func(rune) bool) ([]byte, error) {
	appendEscapedASCII := func(dst []byte, c byte) []byte {
		switch c {
		case '"', '\\':
			dst = append(dst, '\\', c)
		case '\b':
			dst = append(dst, "\\b"...)
		case '\f':
			dst = append(dst, "\\f"...)
		case '\n':
			dst = append(dst, "\\n"...)
		case '\r':
			dst = append(dst, "\\r"...)
		case '\t':
			dst = append(dst, "\\t"...)
		default:
			dst = append(dst, "\\u"...)
			dst = appendHexUint16(dst, uint16(c))
		}
		return dst
	}
	appendEscapedUnicode := func(dst []byte, r rune) []byte {
		if r1, r2 := utf16.EncodeRune(r); r1 != '\ufffd' && r2 != '\ufffd' {
			dst = append(dst, "\\u"...)
			dst = appendHexUint16(dst, uint16(r1))
			dst = append(dst, "\\u"...)
			dst = appendHexUint16(dst, uint16(r2))
		} else {
			dst = append(dst, "\\u"...)
			dst = appendHexUint16(dst, uint16(r))
		}
		return dst
	}

	// Optimize for when escapeRune is nil.
	if escapeRune == nil {
		var i, n int
		dst = append(dst, '"')
		for uint(len(src)) > uint(n) {
			// Handle single-byte ASCII.
			if c := src[n]; c < utf8.RuneSelf {
				n++
				if c < ' ' || c == '"' || c == '\\' {
					dst = append(dst, src[i:n-1]...)
					dst = appendEscapedASCII(dst, c)
					i = n
				}
				continue
			}

			// Handle multi-byte Unicode.
			_, rn := utf8.DecodeRuneInString(src[n:])
			n += rn
			if rn == 1 { // must be utf8.RuneError since we already checked for single-byte ASCII
				dst = append(dst, src[i:n-rn]...)
				if validateUTF8 {
					return dst, &SyntacticError{str: "invalid UTF-8 within string"}
				}
				dst = append(dst, "\ufffd"...)
				i = n
			}
		}
		dst = append(dst, src[i:n]...)
		dst = append(dst, '"')
		return dst, nil
	}

	// Slower implementation for when escapeRune is non-nil.
	var i, n int
	dst = append(dst, '"')
	for uint(len(src)) > uint(n) {
		switch r, rn := utf8.DecodeRuneInString(src[n:]); {
		case r == utf8.RuneError && rn == 1:
			dst = append(dst, src[i:n]...)
			if validateUTF8 {
				return dst, &SyntacticError{str: "invalid UTF-8 within string"}
			}
			if escapeRune('\ufffd') {
				dst = append(dst, `\ufffd`...)
			} else {
				dst = append(dst, "\ufffd"...)
			}
			n += rn
			i = n
		case escapeRune(r):
			dst = append(dst, src[i:n]...)
			dst = appendEscapedUnicode(dst, r)
			n += rn
			i = n
		case r < ' ' || r == '"' || r == '\\':
			dst = append(dst, src[i:n]...)
			dst = appendEscapedASCII(dst, byte(r))
			n += rn
			i = n
		default:
			n += rn
		}
	}
	dst = append(dst, src[i:n]...)
	dst = append(dst, '"')
	return dst, nil
}

// reformatString consumes a JSON string from src and appends it to dst,
// reformatting it if necessary for the given escapeRune parameter.
// It returns the appended output and the remainder of the input.
func reformatString(dst, src []byte, validateUTF8, preserveRaw bool, escapeRune func(rune) bool) ([]byte, []byte, error) {
	// TODO: Should this update valueFlags as input?
	var flags valueFlags
	n, err := consumeString(&flags, src, validateUTF8)
	if err != nil {
		return dst, src[n:], err
	}
	if preserveRaw || (escapeRune == nil && flags.isCanonical()) {
		dst = append(dst, src[:n]...) // copy the string verbatim
		return dst, src[n:], nil
	}

	// TODO: Implement a direct, raw-to-raw reformat for strings.
	// If the escapeRune option would have resulted in no changes to the output,
	// it would be faster to simply append src to dst without going through
	// an intermediary representation in a separate buffer.
	b, _ := unescapeString(make([]byte, 0, n), src[:n])
	dst, _ = appendString(dst, string(b), validateUTF8, escapeRune)
	return dst, src[n:], nil
}

// appendNumber appends src to dst as a JSON number per RFC 7159, section 6.
// It formats numbers similar to the ES6 number-to-string conversion.
// See https://go.dev/issue/14135.
//
// The output is identical to ECMA-262, 6th edition, section 7.1.12.1 and with
// RFC 8785, section 3.2.2.3 for 64-bit floating-point numbers except for -0,
// which is formatted as -0 instead of just 0.
//
// For 32-bit floating-point numbers,
// the output is a 32-bit equivalent of the algorithm.
// Note that ECMA-262 specifies no algorithm for 32-bit numbers.
func appendNumber(dst []byte, src float64, bits int) []byte {
	if bits == 32 {
		src = float64(float32(src))
	}

	abs := math.Abs(src)
	fmt := byte('f')
	if abs != 0 {
		if bits == 64 && (float64(abs) < 1e-6 || float64(abs) >= 1e21) ||
			bits == 32 && (float32(abs) < 1e-6 || float32(abs) >= 1e21) {
			fmt = 'e'
		}
	}
	dst = strconv.AppendFloat(dst, src, fmt, -1, bits)
	if fmt == 'e' {
		// Clean up e-09 to e-9.
		n := len(dst)
		if n >= 4 && dst[n-4] == 'e' && dst[n-3] == '-' && dst[n-2] == '0' {
			dst[n-2] = dst[n-1]
			dst = dst[:n-1]
		}
	}
	return dst
}

// reformatNumber consumes a JSON string from src and appends it to dst,
// canonicalizing it if specified.
// It returns the appended output and the remainder of the input.
func reformatNumber(dst, src []byte, canonicalize bool) ([]byte, []byte, error) {
	n, err := consumeNumber(src)
	if err != nil {
		return dst, src[n:], err
	}
	if !canonicalize {
		dst = append(dst, src[:n]...) // copy the number verbatim
		return dst, src[n:], nil
	}

	// Canonicalize the number per RFC 8785, section 3.2.2.3.
	// As an optimization, we can copy integer numbers below 2⁵³ verbatim.
	const maxExactIntegerDigits = 16 // len(strconv.AppendUint(nil, 1<<53, 10))
	if n < maxExactIntegerDigits && consumeSimpleNumber(src[:n]) == n {
		dst = append(dst, src[:n]...) // copy the number verbatim
		return dst, src[n:], nil
	}
	fv, _ := strconv.ParseFloat(string(src[:n]), 64)
	switch {
	case fv == 0:
		fv = 0 // normalize negative zero as just zero
	case math.IsInf(fv, +1):
		fv = +math.MaxFloat64
	case math.IsInf(fv, -1):
		fv = -math.MaxFloat64
	}
	return appendNumber(dst, fv, 64), src[n:], nil
}

// appendHexUint16 appends src to dst as a 4-byte hexadecimal number.
func appendHexUint16(dst []byte, src uint16) []byte {
	dst = append(dst, "0000"[1+(bits.Len16(src)-1)/4:]...)
	dst = strconv.AppendUint(dst, uint64(src), 16)
	return dst
}
