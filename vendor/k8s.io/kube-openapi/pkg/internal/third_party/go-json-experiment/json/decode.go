// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"errors"
	"io"
	"math"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

// NOTE: The logic for decoding is complicated by the fact that reading from
// an io.Reader into a temporary buffer means that the buffer may contain a
// truncated portion of some valid input, requiring the need to fetch more data.
//
// This file is structured in the following way:
//
//   - consumeXXX functions parse an exact JSON token from a []byte.
//     If the buffer appears truncated, then it returns io.ErrUnexpectedEOF.
//     The consumeSimpleXXX functions are so named because they only handle
//     a subset of the grammar for the JSON token being parsed.
//     They do not handle the full grammar to keep these functions inlineable.
//
//   - Decoder.consumeXXX methods parse the next JSON token from Decoder.buf,
//     automatically fetching more input if necessary. These methods take
//     a position relative to the start of Decoder.buf as an argument and
//     return the end of the consumed JSON token as a position,
//     also relative to the start of Decoder.buf.
//
//   - In the event of an I/O errors or state machine violations,
//     the implementation avoids mutating the state of Decoder
//     (aside from the book-keeping needed to implement Decoder.fetch).
//     For this reason, only Decoder.ReadToken and Decoder.ReadValue are
//     responsible for updated Decoder.prevStart and Decoder.prevEnd.
//
//   - For performance, much of the implementation uses the pattern of calling
//     the inlineable consumeXXX functions first, and if more work is necessary,
//     then it calls the slower Decoder.consumeXXX methods.
//     TODO: Revisit this pattern if the Go compiler provides finer control
//     over exactly which calls are inlined or not.

// DecodeOptions configures how JSON decoding operates.
// The zero value is equivalent to the default settings,
// which is compliant with both RFC 7493 and RFC 8259.
type DecodeOptions struct {
	requireKeyedLiterals
	nonComparable

	// AllowDuplicateNames specifies that JSON objects may contain
	// duplicate member names. Disabling the duplicate name check may provide
	// computational and performance benefits, but breaks compliance with
	// RFC 7493, section 2.3. The input will still be compliant with RFC 8259,
	// which leaves the handling of duplicate names as unspecified behavior.
	AllowDuplicateNames bool

	// AllowInvalidUTF8 specifies that JSON strings may contain invalid UTF-8,
	// which will be mangled as the Unicode replacement character, U+FFFD.
	// This causes the decoder to break compliance with
	// RFC 7493, section 2.1, and RFC 8259, section 8.1.
	AllowInvalidUTF8 bool
}

// Decoder is a streaming decoder for raw JSON tokens and values.
// It is used to read a stream of top-level JSON values,
// each separated by optional whitespace characters.
//
// ReadToken and ReadValue calls may be interleaved.
// For example, the following JSON value:
//
//	{"name":"value","array":[null,false,true,3.14159],"object":{"k":"v"}}
//
// can be parsed with the following calls (ignoring errors for brevity):
//
//	d.ReadToken() // {
//	d.ReadToken() // "name"
//	d.ReadToken() // "value"
//	d.ReadValue() // "array"
//	d.ReadToken() // [
//	d.ReadToken() // null
//	d.ReadToken() // false
//	d.ReadValue() // true
//	d.ReadToken() // 3.14159
//	d.ReadToken() // ]
//	d.ReadValue() // "object"
//	d.ReadValue() // {"k":"v"}
//	d.ReadToken() // }
//
// The above is one of many possible sequence of calls and
// may not represent the most sensible method to call for any given token/value.
// For example, it is probably more common to call ReadToken to obtain a
// string token for object names.
type Decoder struct {
	state
	decodeBuffer
	options DecodeOptions

	stringCache *stringCache // only used when unmarshaling
}

// decodeBuffer is a buffer split into 4 segments:
//
//   - buf[0:prevEnd]         // already read portion of the buffer
//   - buf[prevStart:prevEnd] // previously read value
//   - buf[prevEnd:len(buf)]  // unread portion of the buffer
//   - buf[len(buf):cap(buf)] // unused portion of the buffer
//
// Invariants:
//
//	0 ≤ prevStart ≤ prevEnd ≤ len(buf) ≤ cap(buf)
type decodeBuffer struct {
	peekPos int   // non-zero if valid offset into buf for start of next token
	peekErr error // implies peekPos is -1

	buf       []byte // may alias rd if it is a bytes.Buffer
	prevStart int
	prevEnd   int

	// baseOffset is added to prevStart and prevEnd to obtain
	// the absolute offset relative to the start of io.Reader stream.
	baseOffset int64

	rd io.Reader
}

// NewDecoder constructs a new streaming decoder reading from r.
//
// If r is a bytes.Buffer, then the decoder parses directly from the buffer
// without first copying the contents to an intermediate buffer.
// Additional writes to the buffer must not occur while the decoder is in use.
func NewDecoder(r io.Reader) *Decoder {
	return DecodeOptions{}.NewDecoder(r)
}

// NewDecoder constructs a new streaming decoder reading from r
// configured with the provided options.
func (o DecodeOptions) NewDecoder(r io.Reader) *Decoder {
	d := new(Decoder)
	o.ResetDecoder(d, r)
	return d
}

// ResetDecoder resets a decoder such that it is reading afresh from r and
// configured with the provided options.
func (o DecodeOptions) ResetDecoder(d *Decoder, r io.Reader) {
	if d == nil {
		panic("json: invalid nil Decoder")
	}
	if r == nil {
		panic("json: invalid nil io.Reader")
	}
	d.reset(nil, r, o)
}

func (d *Decoder) reset(b []byte, r io.Reader, o DecodeOptions) {
	d.state.reset()
	d.decodeBuffer = decodeBuffer{buf: b, rd: r}
	d.options = o
}

// Reset resets a decoder such that it is reading afresh from r but
// keep any pre-existing decoder options.
func (d *Decoder) Reset(r io.Reader) {
	d.options.ResetDecoder(d, r)
}

var errBufferWriteAfterNext = errors.New("invalid bytes.Buffer.Write call after calling bytes.Buffer.Next")

// fetch reads at least 1 byte from the underlying io.Reader.
// It returns io.ErrUnexpectedEOF if zero bytes were read and io.EOF was seen.
func (d *Decoder) fetch() error {
	if d.rd == nil {
		return io.ErrUnexpectedEOF
	}

	// Inform objectNameStack that we are about to fetch new buffer content.
	d.names.copyQuotedBuffer(d.buf)

	// Specialize bytes.Buffer for better performance.
	if bb, ok := d.rd.(*bytes.Buffer); ok {
		switch {
		case bb.Len() == 0:
			return io.ErrUnexpectedEOF
		case len(d.buf) == 0:
			d.buf = bb.Next(bb.Len()) // "read" all data in the buffer
			return nil
		default:
			// This only occurs if a partially filled bytes.Buffer was provided
			// and more data is written to it while Decoder is reading from it.
			// This practice will lead to data corruption since future writes
			// may overwrite the contents of the current buffer.
			//
			// The user is trying to use a bytes.Buffer as a pipe,
			// but a bytes.Buffer is poor implementation of a pipe,
			// the purpose-built io.Pipe should be used instead.
			return &ioError{action: "read", err: errBufferWriteAfterNext}
		}
	}

	// Allocate initial buffer if empty.
	if cap(d.buf) == 0 {
		d.buf = make([]byte, 0, 64)
	}

	// Check whether to grow the buffer.
	const maxBufferSize = 4 << 10
	const growthSizeFactor = 2 // higher value is faster
	const growthRateFactor = 2 // higher value is slower
	// By default, grow if below the maximum buffer size.
	grow := cap(d.buf) <= maxBufferSize/growthSizeFactor
	// Growing can be expensive, so only grow
	// if a sufficient number of bytes have been processed.
	grow = grow && int64(cap(d.buf)) < d.previousOffsetEnd()/growthRateFactor
	// If prevStart==0, then fetch was called in order to fetch more data
	// to finish consuming a large JSON value contiguously.
	// Grow if less than 25% of the remaining capacity is available.
	// Note that this may cause the input buffer to exceed maxBufferSize.
	grow = grow || (d.prevStart == 0 && len(d.buf) >= 3*cap(d.buf)/4)

	if grow {
		// Allocate a new buffer and copy the contents of the old buffer over.
		// TODO: Provide a hard limit on the maximum internal buffer size?
		buf := make([]byte, 0, cap(d.buf)*growthSizeFactor)
		d.buf = append(buf, d.buf[d.prevStart:]...)
	} else {
		// Move unread portion of the data to the front.
		n := copy(d.buf[:cap(d.buf)], d.buf[d.prevStart:])
		d.buf = d.buf[:n]
	}
	d.baseOffset += int64(d.prevStart)
	d.prevEnd -= d.prevStart
	d.prevStart = 0

	// Read more data into the internal buffer.
	for {
		n, err := d.rd.Read(d.buf[len(d.buf):cap(d.buf)])
		switch {
		case n > 0:
			d.buf = d.buf[:len(d.buf)+n]
			return nil // ignore errors if any bytes are read
		case err == io.EOF:
			return io.ErrUnexpectedEOF
		case err != nil:
			return &ioError{action: "read", err: err}
		default:
			continue // Read returned (0, nil)
		}
	}
}

const invalidateBufferByte = '#' // invalid starting character for JSON grammar

// invalidatePreviousRead invalidates buffers returned by Peek and Read calls
// so that the first byte is an invalid character.
// This Hyrum-proofs the API against faulty application code that assumes
// values returned by ReadValue remain valid past subsequent Read calls.
func (d *decodeBuffer) invalidatePreviousRead() {
	// Avoid mutating the buffer if d.rd is nil which implies that d.buf
	// is provided by the user code and may not expect mutations.
	isBytesBuffer := func(r io.Reader) bool {
		_, ok := r.(*bytes.Buffer)
		return ok
	}
	if d.rd != nil && !isBytesBuffer(d.rd) && d.prevStart < d.prevEnd && uint(d.prevStart) < uint(len(d.buf)) {
		d.buf[d.prevStart] = invalidateBufferByte
		d.prevStart = d.prevEnd
	}
}

// needMore reports whether there are no more unread bytes.
func (d *decodeBuffer) needMore(pos int) bool {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	return pos == len(d.buf)
}

// injectSyntacticErrorWithPosition wraps a SyntacticError with the position,
// otherwise it returns the error as is.
// It takes a position relative to the start of the start of d.buf.
func (d *decodeBuffer) injectSyntacticErrorWithPosition(err error, pos int) error {
	if serr, ok := err.(*SyntacticError); ok {
		return serr.withOffset(d.baseOffset + int64(pos))
	}
	return err
}

func (d *decodeBuffer) previousOffsetStart() int64 { return d.baseOffset + int64(d.prevStart) }
func (d *decodeBuffer) previousOffsetEnd() int64   { return d.baseOffset + int64(d.prevEnd) }
func (d *decodeBuffer) previousBuffer() []byte     { return d.buf[d.prevStart:d.prevEnd] }
func (d *decodeBuffer) unreadBuffer() []byte       { return d.buf[d.prevEnd:len(d.buf)] }

// PeekKind retrieves the next token kind, but does not advance the read offset.
// It returns 0 if there are no more tokens.
func (d *Decoder) PeekKind() Kind {
	// Check whether we have a cached peek result.
	if d.peekPos > 0 {
		return Kind(d.buf[d.peekPos]).normalize()
	}

	var err error
	d.invalidatePreviousRead()
	pos := d.prevEnd

	// Consume leading whitespace.
	pos += consumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			if err == io.ErrUnexpectedEOF && d.tokens.depth() == 1 {
				err = io.EOF // EOF possibly if no Tokens present after top-level value
			}
			d.peekPos, d.peekErr = -1, err
			return invalidKind
		}
	}

	// Consume colon or comma.
	var delim byte
	if c := d.buf[pos]; c == ':' || c == ',' {
		delim = c
		pos += 1
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				d.peekPos, d.peekErr = -1, err
				return invalidKind
			}
		}
	}
	next := Kind(d.buf[pos]).normalize()
	if d.tokens.needDelim(next) != delim {
		pos = d.prevEnd // restore position to right after leading whitespace
		pos += consumeWhitespace(d.buf[pos:])
		err = d.tokens.checkDelim(delim, next)
		err = d.injectSyntacticErrorWithPosition(err, pos)
		d.peekPos, d.peekErr = -1, err
		return invalidKind
	}

	// This may set peekPos to zero, which is indistinguishable from
	// the uninitialized state. While a small hit to performance, it is correct
	// since ReadValue and ReadToken will disregard the cached result and
	// recompute the next kind.
	d.peekPos, d.peekErr = pos, nil
	return next
}

// SkipValue is semantically equivalent to calling ReadValue and discarding
// the result except that memory is not wasted trying to hold the entire result.
func (d *Decoder) SkipValue() error {
	switch d.PeekKind() {
	case '{', '[':
		// For JSON objects and arrays, keep skipping all tokens
		// until the depth matches the starting depth.
		depth := d.tokens.depth()
		for {
			if _, err := d.ReadToken(); err != nil {
				return err
			}
			if depth >= d.tokens.depth() {
				return nil
			}
		}
	default:
		// Trying to skip a value when the next token is a '}' or ']'
		// will result in an error being returned here.
		if _, err := d.ReadValue(); err != nil {
			return err
		}
		return nil
	}
}

// ReadToken reads the next Token, advancing the read offset.
// The returned token is only valid until the next Peek, Read, or Skip call.
// It returns io.EOF if there are no more tokens.
func (d *Decoder) ReadToken() (Token, error) {
	// Determine the next kind.
	var err error
	var next Kind
	pos := d.peekPos
	if pos != 0 {
		// Use cached peek result.
		if d.peekErr != nil {
			err := d.peekErr
			d.peekPos, d.peekErr = 0, nil // possibly a transient I/O error
			return Token{}, err
		}
		next = Kind(d.buf[pos]).normalize()
		d.peekPos = 0 // reset cache
	} else {
		d.invalidatePreviousRead()
		pos = d.prevEnd

		// Consume leading whitespace.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				if err == io.ErrUnexpectedEOF && d.tokens.depth() == 1 {
					err = io.EOF // EOF possibly if no Tokens present after top-level value
				}
				return Token{}, err
			}
		}

		// Consume colon or comma.
		var delim byte
		if c := d.buf[pos]; c == ':' || c == ',' {
			delim = c
			pos += 1
			pos += consumeWhitespace(d.buf[pos:])
			if d.needMore(pos) {
				if pos, err = d.consumeWhitespace(pos); err != nil {
					return Token{}, err
				}
			}
		}
		next = Kind(d.buf[pos]).normalize()
		if d.tokens.needDelim(next) != delim {
			pos = d.prevEnd // restore position to right after leading whitespace
			pos += consumeWhitespace(d.buf[pos:])
			err = d.tokens.checkDelim(delim, next)
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
		}
	}

	// Handle the next token.
	var n int
	switch next {
	case 'n':
		if consumeNull(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "null")
			if err != nil {
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
			}
		} else {
			pos += len("null")
		}
		if err = d.tokens.appendLiteral(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos-len("null")) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return Null, nil

	case 'f':
		if consumeFalse(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "false")
			if err != nil {
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
			}
		} else {
			pos += len("false")
		}
		if err = d.tokens.appendLiteral(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos-len("false")) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return False, nil

	case 't':
		if consumeTrue(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "true")
			if err != nil {
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
			}
		} else {
			pos += len("true")
		}
		if err = d.tokens.appendLiteral(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos-len("true")) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return True, nil

	case '"':
		var flags valueFlags // TODO: Preserve this in Token?
		if n = consumeSimpleString(d.buf[pos:]); n == 0 {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeString(&flags, pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			if err != nil {
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
			}
		} else {
			pos += n
		}
		if !d.options.AllowDuplicateNames && d.tokens.last.needObjectName() {
			if !d.tokens.last.isValidNamespace() {
				return Token{}, errInvalidNamespace
			}
			if d.tokens.last.isActiveNamespace() && !d.namespaces.last().insertQuoted(d.buf[pos-n:pos], flags.isVerbatim()) {
				err = &SyntacticError{str: "duplicate name " + string(d.buf[pos-n:pos]) + " in object"}
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos-n) // report position at start of string
			}
			d.names.replaceLastQuotedOffset(pos - n) // only replace if insertQuoted succeeds
		}
		if err = d.tokens.appendString(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos-n) // report position at start of string
		}
		d.prevStart, d.prevEnd = pos-n, pos
		return Token{raw: &d.decodeBuffer, num: uint64(d.previousOffsetStart())}, nil

	case '0':
		// NOTE: Since JSON numbers are not self-terminating,
		// we need to make sure that the next byte is not part of a number.
		if n = consumeSimpleNumber(d.buf[pos:]); n == 0 || d.needMore(pos+n) {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeNumber(pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			if err != nil {
				return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
			}
		} else {
			pos += n
		}
		if err = d.tokens.appendNumber(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos-n) // report position at start of number
		}
		d.prevStart, d.prevEnd = pos-n, pos
		return Token{raw: &d.decodeBuffer, num: uint64(d.previousOffsetStart())}, nil

	case '{':
		if err = d.tokens.pushObject(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
		}
		if !d.options.AllowDuplicateNames {
			d.names.push()
			d.namespaces.push()
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return ObjectStart, nil

	case '}':
		if err = d.tokens.popObject(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
		}
		if !d.options.AllowDuplicateNames {
			d.names.pop()
			d.namespaces.pop()
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return ObjectEnd, nil

	case '[':
		if err = d.tokens.pushArray(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return ArrayStart, nil

	case ']':
		if err = d.tokens.popArray(); err != nil {
			return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return ArrayEnd, nil

	default:
		err = newInvalidCharacterError(d.buf[pos:], "at start of token")
		return Token{}, d.injectSyntacticErrorWithPosition(err, pos)
	}
}

type valueFlags uint

const (
	_ valueFlags = (1 << iota) / 2 // powers of two starting with zero

	stringNonVerbatim  // string cannot be naively treated as valid UTF-8
	stringNonCanonical // string not formatted according to RFC 8785, section 3.2.2.2.
	// TODO: Track whether a number is a non-integer?
)

func (f *valueFlags) set(f2 valueFlags) { *f |= f2 }
func (f valueFlags) isVerbatim() bool   { return f&stringNonVerbatim == 0 }
func (f valueFlags) isCanonical() bool  { return f&stringNonCanonical == 0 }

// ReadValue returns the next raw JSON value, advancing the read offset.
// The value is stripped of any leading or trailing whitespace.
// The returned value is only valid until the next Peek, Read, or Skip call and
// may not be mutated while the Decoder remains in use.
// If the decoder is currently at the end token for an object or array,
// then it reports a SyntacticError and the internal state remains unchanged.
// It returns io.EOF if there are no more values.
func (d *Decoder) ReadValue() (RawValue, error) {
	var flags valueFlags
	return d.readValue(&flags)
}
func (d *Decoder) readValue(flags *valueFlags) (RawValue, error) {
	// Determine the next kind.
	var err error
	var next Kind
	pos := d.peekPos
	if pos != 0 {
		// Use cached peek result.
		if d.peekErr != nil {
			err := d.peekErr
			d.peekPos, d.peekErr = 0, nil // possibly a transient I/O error
			return nil, err
		}
		next = Kind(d.buf[pos]).normalize()
		d.peekPos = 0 // reset cache
	} else {
		d.invalidatePreviousRead()
		pos = d.prevEnd

		// Consume leading whitespace.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				if err == io.ErrUnexpectedEOF && d.tokens.depth() == 1 {
					err = io.EOF // EOF possibly if no Tokens present after top-level value
				}
				return nil, err
			}
		}

		// Consume colon or comma.
		var delim byte
		if c := d.buf[pos]; c == ':' || c == ',' {
			delim = c
			pos += 1
			pos += consumeWhitespace(d.buf[pos:])
			if d.needMore(pos) {
				if pos, err = d.consumeWhitespace(pos); err != nil {
					return nil, err
				}
			}
		}
		next = Kind(d.buf[pos]).normalize()
		if d.tokens.needDelim(next) != delim {
			pos = d.prevEnd // restore position to right after leading whitespace
			pos += consumeWhitespace(d.buf[pos:])
			err = d.tokens.checkDelim(delim, next)
			return nil, d.injectSyntacticErrorWithPosition(err, pos)
		}
	}

	// Handle the next value.
	oldAbsPos := d.baseOffset + int64(pos)
	pos, err = d.consumeValue(flags, pos)
	newAbsPos := d.baseOffset + int64(pos)
	n := int(newAbsPos - oldAbsPos)
	if err != nil {
		return nil, d.injectSyntacticErrorWithPosition(err, pos)
	}
	switch next {
	case 'n', 't', 'f':
		err = d.tokens.appendLiteral()
	case '"':
		if !d.options.AllowDuplicateNames && d.tokens.last.needObjectName() {
			if !d.tokens.last.isValidNamespace() {
				err = errInvalidNamespace
				break
			}
			if d.tokens.last.isActiveNamespace() && !d.namespaces.last().insertQuoted(d.buf[pos-n:pos], flags.isVerbatim()) {
				err = &SyntacticError{str: "duplicate name " + string(d.buf[pos-n:pos]) + " in object"}
				break
			}
			d.names.replaceLastQuotedOffset(pos - n) // only replace if insertQuoted succeeds
		}
		err = d.tokens.appendString()
	case '0':
		err = d.tokens.appendNumber()
	case '{':
		if err = d.tokens.pushObject(); err != nil {
			break
		}
		if err = d.tokens.popObject(); err != nil {
			panic("BUG: popObject should never fail immediately after pushObject: " + err.Error())
		}
	case '[':
		if err = d.tokens.pushArray(); err != nil {
			break
		}
		if err = d.tokens.popArray(); err != nil {
			panic("BUG: popArray should never fail immediately after pushArray: " + err.Error())
		}
	}
	if err != nil {
		return nil, d.injectSyntacticErrorWithPosition(err, pos-n) // report position at start of value
	}
	d.prevEnd = pos
	d.prevStart = pos - n
	return d.buf[pos-n : pos : pos], nil
}

// checkEOF verifies that the input has no more data.
func (d *Decoder) checkEOF() error {
	switch pos, err := d.consumeWhitespace(d.prevEnd); err {
	case nil:
		return newInvalidCharacterError(d.buf[pos:], "after top-level value")
	case io.ErrUnexpectedEOF:
		return nil
	default:
		return err
	}
}

// consumeWhitespace consumes all whitespace starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the last whitespace.
// If it returns nil, there is guaranteed to at least be one unread byte.
//
// The following pattern is common in this implementation:
//
//	pos += consumeWhitespace(d.buf[pos:])
//	if d.needMore(pos) {
//		if pos, err = d.consumeWhitespace(pos); err != nil {
//			return ...
//		}
//	}
//
// It is difficult to simplify this without sacrificing performance since
// consumeWhitespace must be inlined. The body of the if statement is
// executed only in rare situations where we need to fetch more data.
// Since fetching may return an error, we also need to check the error.
func (d *Decoder) consumeWhitespace(pos int) (newPos int, err error) {
	for {
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos, err
			}
			continue
		}
		return pos, nil
	}
}

// consumeValue consumes a single JSON value starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the value.
func (d *Decoder) consumeValue(flags *valueFlags, pos int) (newPos int, err error) {
	for {
		var n int
		var err error
		switch next := Kind(d.buf[pos]).normalize(); next {
		case 'n':
			if n = consumeNull(d.buf[pos:]); n == 0 {
				n, err = consumeLiteral(d.buf[pos:], "null")
			}
		case 'f':
			if n = consumeFalse(d.buf[pos:]); n == 0 {
				n, err = consumeLiteral(d.buf[pos:], "false")
			}
		case 't':
			if n = consumeTrue(d.buf[pos:]); n == 0 {
				n, err = consumeLiteral(d.buf[pos:], "true")
			}
		case '"':
			if n = consumeSimpleString(d.buf[pos:]); n == 0 {
				return d.consumeString(flags, pos)
			}
		case '0':
			// NOTE: Since JSON numbers are not self-terminating,
			// we need to make sure that the next byte is not part of a number.
			if n = consumeSimpleNumber(d.buf[pos:]); n == 0 || d.needMore(pos+n) {
				return d.consumeNumber(pos)
			}
		case '{':
			return d.consumeObject(flags, pos)
		case '[':
			return d.consumeArray(flags, pos)
		default:
			return pos, newInvalidCharacterError(d.buf[pos:], "at start of value")
		}
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeLiteral consumes a single JSON literal starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the literal.
func (d *Decoder) consumeLiteral(pos int, lit string) (newPos int, err error) {
	for {
		n, err := consumeLiteral(d.buf[pos:], lit)
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeString consumes a single JSON string starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the string.
func (d *Decoder) consumeString(flags *valueFlags, pos int) (newPos int, err error) {
	var n int
	for {
		n, err = consumeStringResumable(flags, d.buf[pos:], n, !d.options.AllowInvalidUTF8)
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeNumber consumes a single JSON number starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the number.
func (d *Decoder) consumeNumber(pos int) (newPos int, err error) {
	var n int
	var state consumeNumberState
	for {
		n, state, err = consumeNumberResumable(d.buf[pos:], n, state)
		// NOTE: Since JSON numbers are not self-terminating,
		// we need to make sure that the next byte is not part of a number.
		if err == io.ErrUnexpectedEOF || d.needMore(pos+n) {
			mayTerminate := err == nil
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				if mayTerminate && err == io.ErrUnexpectedEOF {
					return pos + n, nil
				}
				return pos, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeObject consumes a single JSON object starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the object.
func (d *Decoder) consumeObject(flags *valueFlags, pos int) (newPos int, err error) {
	var n int
	var names *objectNamespace
	if !d.options.AllowDuplicateNames {
		d.namespaces.push()
		defer d.namespaces.pop()
		names = d.namespaces.last()
	}

	// Handle before start.
	if d.buf[pos] != '{' {
		panic("BUG: consumeObject must be called with a buffer that starts with '{'")
	}
	pos++

	// Handle after start.
	pos += consumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			return pos, err
		}
	}
	if d.buf[pos] == '}' {
		pos++
		return pos, nil
	}

	for {
		// Handle before name.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		var flags2 valueFlags
		if n = consumeSimpleString(d.buf[pos:]); n == 0 {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeString(&flags2, pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			flags.set(flags2)
			if err != nil {
				return pos, err
			}
		} else {
			pos += n
		}
		if !d.options.AllowDuplicateNames && !names.insertQuoted(d.buf[pos-n:pos], flags2.isVerbatim()) {
			return pos - n, &SyntacticError{str: "duplicate name " + string(d.buf[pos-n:pos]) + " in object"}
		}

		// Handle after name.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		if d.buf[pos] != ':' {
			return pos, newInvalidCharacterError(d.buf[pos:], "after object name (expecting ':')")
		}
		pos++

		// Handle before value.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		pos, err = d.consumeValue(flags, pos)
		if err != nil {
			return pos, err
		}

		// Handle after value.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		switch d.buf[pos] {
		case ',':
			pos++
			continue
		case '}':
			pos++
			return pos, nil
		default:
			return pos, newInvalidCharacterError(d.buf[pos:], "after object value (expecting ',' or '}')")
		}
	}
}

// consumeArray consumes a single JSON array starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the array.
func (d *Decoder) consumeArray(flags *valueFlags, pos int) (newPos int, err error) {
	// Handle before start.
	if d.buf[pos] != '[' {
		panic("BUG: consumeArray must be called with a buffer that starts with '['")
	}
	pos++

	// Handle after start.
	pos += consumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			return pos, err
		}
	}
	if d.buf[pos] == ']' {
		pos++
		return pos, nil
	}

	for {
		// Handle before value.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		pos, err = d.consumeValue(flags, pos)
		if err != nil {
			return pos, err
		}

		// Handle after value.
		pos += consumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		switch d.buf[pos] {
		case ',':
			pos++
			continue
		case ']':
			pos++
			return pos, nil
		default:
			return pos, newInvalidCharacterError(d.buf[pos:], "after array value (expecting ',' or ']')")
		}
	}
}

// InputOffset returns the current input byte offset. It gives the location
// of the next byte immediately after the most recently returned token or value.
// The number of bytes actually read from the underlying io.Reader may be more
// than this offset due to internal buffering effects.
func (d *Decoder) InputOffset() int64 {
	return d.previousOffsetEnd()
}

// UnreadBuffer returns the data remaining in the unread buffer,
// which may contain zero or more bytes.
// The returned buffer must not be mutated while Decoder continues to be used.
// The buffer contents are valid until the next Peek, Read, or Skip call.
func (d *Decoder) UnreadBuffer() []byte {
	return d.unreadBuffer()
}

// StackDepth returns the depth of the state machine for read JSON data.
// Each level on the stack represents a nested JSON object or array.
// It is incremented whenever an ObjectStart or ArrayStart token is encountered
// and decremented whenever an ObjectEnd or ArrayEnd token is encountered.
// The depth is zero-indexed, where zero represents the top-level JSON value.
func (d *Decoder) StackDepth() int {
	// NOTE: Keep in sync with Encoder.StackDepth.
	return d.tokens.depth() - 1
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
func (d *Decoder) StackIndex(i int) (Kind, int) {
	// NOTE: Keep in sync with Encoder.StackIndex.
	switch s := d.tokens.index(i); {
	case i > 0 && s.isObject():
		return '{', s.length()
	case i > 0 && s.isArray():
		return '[', s.length()
	default:
		return 0, s.length()
	}
}

// StackPointer returns a JSON Pointer (RFC 6901) to the most recently read value.
// Object names are only present if AllowDuplicateNames is false, otherwise
// object members are represented using their index within the object.
func (d *Decoder) StackPointer() string {
	d.names.copyQuotedBuffer(d.buf)
	return string(d.appendStackPointer(nil))
}

// consumeWhitespace consumes leading JSON whitespace per RFC 7159, section 2.
func consumeWhitespace(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	for len(b) > n && (b[n] == ' ' || b[n] == '\t' || b[n] == '\r' || b[n] == '\n') {
		n++
	}
	return n
}

// consumeNull consumes the next JSON null literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func consumeNull(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	const literal = "null"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// consumeFalse consumes the next JSON false literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func consumeFalse(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	const literal = "false"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// consumeTrue consumes the next JSON true literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func consumeTrue(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	const literal = "true"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// consumeLiteral consumes the next JSON literal per RFC 7159, section 3.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
func consumeLiteral(b []byte, lit string) (n int, err error) {
	for i := 0; i < len(b) && i < len(lit); i++ {
		if b[i] != lit[i] {
			return i, newInvalidCharacterError(b[i:], "within literal "+lit+" (expecting "+strconv.QuoteRune(rune(lit[i]))+")")
		}
	}
	if len(b) < len(lit) {
		return len(b), io.ErrUnexpectedEOF
	}
	return len(lit), nil
}

// consumeSimpleString consumes the next JSON string per RFC 7159, section 7
// but is limited to the grammar for an ASCII string without escape sequences.
// It returns 0 if it is invalid or more complicated than a simple string,
// in which case consumeString should be called.
func consumeSimpleString(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	if len(b) > 0 && b[0] == '"' {
		n++
		for len(b) > n && (' ' <= b[n] && b[n] != '\\' && b[n] != '"' && b[n] < utf8.RuneSelf) {
			n++
		}
		if len(b) > n && b[n] == '"' {
			n++
			return n
		}
	}
	return 0
}

// consumeString consumes the next JSON string per RFC 7159, section 7.
// If validateUTF8 is false, then this allows the presence of invalid UTF-8
// characters within the string itself.
// It reports the number of bytes consumed and whether an error was encountered.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
func consumeString(flags *valueFlags, b []byte, validateUTF8 bool) (n int, err error) {
	return consumeStringResumable(flags, b, 0, validateUTF8)
}

// consumeStringResumable is identical to consumeString but supports resuming
// from a previous call that returned io.ErrUnexpectedEOF.
func consumeStringResumable(flags *valueFlags, b []byte, resumeOffset int, validateUTF8 bool) (n int, err error) {
	// Consume the leading double quote.
	switch {
	case resumeOffset > 0:
		n = resumeOffset // already handled the leading quote
	case uint(len(b)) == 0:
		return n, io.ErrUnexpectedEOF
	case b[0] == '"':
		n++
	default:
		return n, newInvalidCharacterError(b[n:], `at start of string (expecting '"')`)
	}

	// Consume every character in the string.
	for uint(len(b)) > uint(n) {
		// Optimize for long sequences of unescaped characters.
		noEscape := func(c byte) bool {
			return c < utf8.RuneSelf && ' ' <= c && c != '\\' && c != '"'
		}
		for uint(len(b)) > uint(n) && noEscape(b[n]) {
			n++
		}
		if uint(len(b)) <= uint(n) {
			return n, io.ErrUnexpectedEOF
		}

		// Check for terminating double quote.
		if b[n] == '"' {
			n++
			return n, nil
		}

		switch r, rn := utf8.DecodeRune(b[n:]); {
		// Handle UTF-8 encoded byte sequence.
		// Due to specialized handling of ASCII above, we know that
		// all normal sequences at this point must be 2 bytes or larger.
		case rn > 1:
			n += rn
		// Handle escape sequence.
		case r == '\\':
			flags.set(stringNonVerbatim)
			resumeOffset = n
			if uint(len(b)) < uint(n+2) {
				return resumeOffset, io.ErrUnexpectedEOF
			}
			switch r := b[n+1]; r {
			case '/':
				// Forward slash is the only character with 3 representations.
				// Per RFC 8785, section 3.2.2.2., this must not be escaped.
				flags.set(stringNonCanonical)
				n += 2
			case '"', '\\', 'b', 'f', 'n', 'r', 't':
				n += 2
			case 'u':
				if uint(len(b)) < uint(n+6) {
					if !hasEscapeSequencePrefix(b[n:]) {
						flags.set(stringNonCanonical)
						return n, &SyntacticError{str: "invalid escape sequence " + strconv.Quote(string(b[n:])) + " within string"}
					}
					return resumeOffset, io.ErrUnexpectedEOF
				}
				v1, ok := parseHexUint16(b[n+2 : n+6])
				if !ok {
					flags.set(stringNonCanonical)
					return n, &SyntacticError{str: "invalid escape sequence " + strconv.Quote(string(b[n:n+6])) + " within string"}
				}
				// Only certain control characters can use the \uFFFF notation
				// for canonical formatting (per RFC 8785, section 3.2.2.2.).
				switch v1 {
				// \uFFFF notation not permitted for these characters.
				case '\b', '\f', '\n', '\r', '\t':
					flags.set(stringNonCanonical)
				default:
					// \uFFFF notation only permitted for control characters.
					if v1 >= ' ' {
						flags.set(stringNonCanonical)
					} else {
						// \uFFFF notation must be lower case.
						for _, c := range b[n+2 : n+6] {
							if 'A' <= c && c <= 'F' {
								flags.set(stringNonCanonical)
							}
						}
					}
				}
				n += 6

				if validateUTF8 && utf16.IsSurrogate(rune(v1)) {
					if uint(len(b)) >= uint(n+2) && (b[n] != '\\' || b[n+1] != 'u') {
						return n, &SyntacticError{str: "invalid unpaired surrogate half within string"}
					}
					if uint(len(b)) < uint(n+6) {
						if !hasEscapeSequencePrefix(b[n:]) {
							flags.set(stringNonCanonical)
							return n, &SyntacticError{str: "invalid escape sequence " + strconv.Quote(string(b[n:])) + " within string"}
						}
						return resumeOffset, io.ErrUnexpectedEOF
					}
					v2, ok := parseHexUint16(b[n+2 : n+6])
					if !ok {
						return n, &SyntacticError{str: "invalid escape sequence " + strconv.Quote(string(b[n:n+6])) + " within string"}
					}
					if utf16.DecodeRune(rune(v1), rune(v2)) == utf8.RuneError {
						return n, &SyntacticError{str: "invalid surrogate pair in string"}
					}
					n += 6
				}
			default:
				flags.set(stringNonCanonical)
				return n, &SyntacticError{str: "invalid escape sequence " + strconv.Quote(string(b[n:n+2])) + " within string"}
			}
		// Handle invalid UTF-8.
		case r == utf8.RuneError:
			if !utf8.FullRune(b[n:]) {
				return n, io.ErrUnexpectedEOF
			}
			flags.set(stringNonVerbatim | stringNonCanonical)
			if validateUTF8 {
				return n, &SyntacticError{str: "invalid UTF-8 within string"}
			}
			n++
		// Handle invalid control characters.
		case r < ' ':
			flags.set(stringNonVerbatim | stringNonCanonical)
			return n, newInvalidCharacterError(b[n:], "within string (expecting non-control character)")
		default:
			panic("BUG: unhandled character " + quoteRune(b[n:]))
		}
	}
	return n, io.ErrUnexpectedEOF
}

// hasEscapeSequencePrefix reports whether b is possibly
// the truncated prefix of a \uFFFF escape sequence.
func hasEscapeSequencePrefix(b []byte) bool {
	for i, c := range b {
		switch {
		case i == 0 && c != '\\':
			return false
		case i == 1 && c != 'u':
			return false
		case i >= 2 && i < 6 && !('0' <= c && c <= '9') && !('a' <= c && c <= 'f') && !('A' <= c && c <= 'F'):
			return false
		}
	}
	return true
}

// unescapeString appends the unescaped form of a JSON string in src to dst.
// Any invalid UTF-8 within the string will be replaced with utf8.RuneError.
// The input must be an entire JSON string with no surrounding whitespace.
func unescapeString(dst, src []byte) (v []byte, ok bool) {
	// Consume leading double quote.
	if uint(len(src)) == 0 || src[0] != '"' {
		return dst, false
	}
	i, n := 1, 1

	// Consume every character until completion.
	for uint(len(src)) > uint(n) {
		// Optimize for long sequences of unescaped characters.
		noEscape := func(c byte) bool {
			return c < utf8.RuneSelf && ' ' <= c && c != '\\' && c != '"'
		}
		for uint(len(src)) > uint(n) && noEscape(src[n]) {
			n++
		}
		if uint(len(src)) <= uint(n) {
			break
		}

		// Check for terminating double quote.
		if src[n] == '"' {
			dst = append(dst, src[i:n]...)
			n++
			return dst, len(src) == n
		}

		switch r, rn := utf8.DecodeRune(src[n:]); {
		// Handle UTF-8 encoded byte sequence.
		// Due to specialized handling of ASCII above, we know that
		// all normal sequences at this point must be 2 bytes or larger.
		case rn > 1:
			n += rn
		// Handle escape sequence.
		case r == '\\':
			dst = append(dst, src[i:n]...)
			if r < ' ' {
				return dst, false // invalid control character or unescaped quote
			}

			// Handle escape sequence.
			if uint(len(src)) < uint(n+2) {
				return dst, false // truncated escape sequence
			}
			switch r := src[n+1]; r {
			case '"', '\\', '/':
				dst = append(dst, r)
				n += 2
			case 'b':
				dst = append(dst, '\b')
				n += 2
			case 'f':
				dst = append(dst, '\f')
				n += 2
			case 'n':
				dst = append(dst, '\n')
				n += 2
			case 'r':
				dst = append(dst, '\r')
				n += 2
			case 't':
				dst = append(dst, '\t')
				n += 2
			case 'u':
				if uint(len(src)) < uint(n+6) {
					return dst, false // truncated escape sequence
				}
				v1, ok := parseHexUint16(src[n+2 : n+6])
				if !ok {
					return dst, false // invalid escape sequence
				}
				n += 6

				// Check whether this is a surrogate half.
				r := rune(v1)
				if utf16.IsSurrogate(r) {
					r = utf8.RuneError // assume failure unless the following succeeds
					if uint(len(src)) >= uint(n+6) && src[n+0] == '\\' && src[n+1] == 'u' {
						if v2, ok := parseHexUint16(src[n+2 : n+6]); ok {
							if r = utf16.DecodeRune(rune(v1), rune(v2)); r != utf8.RuneError {
								n += 6
							}
						}
					}
				}

				dst = utf8.AppendRune(dst, r)
			default:
				return dst, false // invalid escape sequence
			}
			i = n
		// Handle invalid UTF-8.
		case r == utf8.RuneError:
			// NOTE: An unescaped string may be longer than the escaped string
			// because invalid UTF-8 bytes are being replaced.
			dst = append(dst, src[i:n]...)
			dst = append(dst, "\uFFFD"...)
			n += rn
			i = n
		// Handle invalid control characters.
		case r < ' ':
			dst = append(dst, src[i:n]...)
			return dst, false // invalid control character or unescaped quote
		default:
			panic("BUG: unhandled character " + quoteRune(src[n:]))
		}
	}
	dst = append(dst, src[i:n]...)
	return dst, false // truncated input
}

// unescapeStringMayCopy returns the unescaped form of b.
// If there are no escaped characters, the output is simply a subslice of
// the input with the surrounding quotes removed.
// Otherwise, a new buffer is allocated for the output.
func unescapeStringMayCopy(b []byte, isVerbatim bool) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	if isVerbatim {
		return b[len(`"`) : len(b)-len(`"`)]
	}
	b, _ = unescapeString(make([]byte, 0, len(b)), b)
	return b
}

// consumeSimpleNumber consumes the next JSON number per RFC 7159, section 6
// but is limited to the grammar for a positive integer.
// It returns 0 if it is invalid or more complicated than a simple integer,
// in which case consumeNumber should be called.
func consumeSimpleNumber(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlineable.
	if len(b) > 0 {
		if b[0] == '0' {
			n++
		} else if '1' <= b[0] && b[0] <= '9' {
			n++
			for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
				n++
			}
		} else {
			return 0
		}
		if len(b) == n || !(b[n] == '.' || b[n] == 'e' || b[n] == 'E') {
			return n
		}
	}
	return 0
}

type consumeNumberState uint

const (
	consumeNumberInit consumeNumberState = iota
	beforeIntegerDigits
	withinIntegerDigits
	beforeFractionalDigits
	withinFractionalDigits
	beforeExponentDigits
	withinExponentDigits
)

// consumeNumber consumes the next JSON number per RFC 7159, section 6.
// It reports the number of bytes consumed and whether an error was encountered.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
//
// Note that JSON numbers are not self-terminating.
// If the entire input is consumed, then the caller needs to consider whether
// there may be subsequent unread data that may still be part of this number.
func consumeNumber(b []byte) (n int, err error) {
	n, _, err = consumeNumberResumable(b, 0, consumeNumberInit)
	return n, err
}

// consumeNumberResumable is identical to consumeNumber but supports resuming
// from a previous call that returned io.ErrUnexpectedEOF.
func consumeNumberResumable(b []byte, resumeOffset int, state consumeNumberState) (n int, _ consumeNumberState, err error) {
	// Jump to the right state when resuming from a partial consumption.
	n = resumeOffset
	if state > consumeNumberInit {
		switch state {
		case withinIntegerDigits, withinFractionalDigits, withinExponentDigits:
			// Consume leading digits.
			for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
				n++
			}
			if len(b) == n {
				return n, state, nil // still within the same state
			}
			state++ // switches "withinX" to "beforeY" where Y is the state after X
		}
		switch state {
		case beforeIntegerDigits:
			goto beforeInteger
		case beforeFractionalDigits:
			goto beforeFractional
		case beforeExponentDigits:
			goto beforeExponent
		default:
			return n, state, nil
		}
	}

	// Consume required integer component (with optional minus sign).
beforeInteger:
	resumeOffset = n
	if len(b) > 0 && b[0] == '-' {
		n++
	}
	switch {
	case len(b) == n:
		return resumeOffset, beforeIntegerDigits, io.ErrUnexpectedEOF
	case b[n] == '0':
		n++
		state = beforeFractionalDigits
	case '1' <= b[n] && b[n] <= '9':
		n++
		for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinIntegerDigits
	default:
		return n, state, newInvalidCharacterError(b[n:], "within number (expecting digit)")
	}

	// Consume optional fractional component.
beforeFractional:
	if len(b) > n && b[n] == '.' {
		resumeOffset = n
		n++
		switch {
		case len(b) == n:
			return resumeOffset, beforeFractionalDigits, io.ErrUnexpectedEOF
		case '0' <= b[n] && b[n] <= '9':
			n++
		default:
			return n, state, newInvalidCharacterError(b[n:], "within number (expecting digit)")
		}
		for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinFractionalDigits
	}

	// Consume optional exponent component.
beforeExponent:
	if len(b) > n && (b[n] == 'e' || b[n] == 'E') {
		resumeOffset = n
		n++
		if len(b) > n && (b[n] == '-' || b[n] == '+') {
			n++
		}
		switch {
		case len(b) == n:
			return resumeOffset, beforeExponentDigits, io.ErrUnexpectedEOF
		case '0' <= b[n] && b[n] <= '9':
			n++
		default:
			return n, state, newInvalidCharacterError(b[n:], "within number (expecting digit)")
		}
		for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinExponentDigits
	}

	return n, state, nil
}

// parseHexUint16 is similar to strconv.ParseUint,
// but operates directly on []byte and is optimized for base-16.
// See https://go.dev/issue/42429.
func parseHexUint16(b []byte) (v uint16, ok bool) {
	if len(b) != 4 {
		return 0, false
	}
	for _, c := range b[:4] {
		switch {
		case '0' <= c && c <= '9':
			c = c - '0'
		case 'a' <= c && c <= 'f':
			c = 10 + c - 'a'
		case 'A' <= c && c <= 'F':
			c = 10 + c - 'A'
		default:
			return 0, false
		}
		v = v*16 + uint16(c)
	}
	return v, true
}

// parseDecUint is similar to strconv.ParseUint,
// but operates directly on []byte and is optimized for base-10.
// If the number is syntactically valid but overflows uint64,
// then it returns (math.MaxUint64, false).
// See https://go.dev/issue/42429.
func parseDecUint(b []byte) (v uint64, ok bool) {
	// Overflow logic is based on strconv/atoi.go:138-149 from Go1.15, where:
	//   - cutoff is equal to math.MaxUint64/10+1, and
	//   - the n1 > maxVal check is unnecessary
	//     since maxVal is equivalent to math.MaxUint64.
	var n int
	var overflow bool
	for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
		overflow = overflow || v >= math.MaxUint64/10+1
		v *= 10

		v1 := v + uint64(b[n]-'0')
		overflow = overflow || v1 < v
		v = v1

		n++
	}
	if n == 0 || len(b) != n {
		return 0, false
	}
	if overflow {
		return math.MaxUint64, false
	}
	return v, true
}

// parseFloat parses a floating point number according to the Go float grammar.
// Note that the JSON number grammar is a strict subset.
//
// If the number overflows the finite representation of a float,
// then we return MaxFloat since any finite value will always be infinitely
// more accurate at representing another finite value than an infinite value.
func parseFloat(b []byte, bits int) (v float64, ok bool) {
	// Fast path for exact integer numbers which fit in the
	// 24-bit or 53-bit significand of a float32 or float64.
	var negLen int // either 0 or 1
	if len(b) > 0 && b[0] == '-' {
		negLen = 1
	}
	u, ok := parseDecUint(b[negLen:])
	if ok && ((bits == 32 && u <= 1<<24) || (bits == 64 && u <= 1<<53)) {
		return math.Copysign(float64(u), float64(-1*negLen)), true
	}

	// Note that the []byte->string conversion unfortunately allocates.
	// See https://go.dev/issue/42429 for more information.
	fv, err := strconv.ParseFloat(string(b), bits)
	if math.IsInf(fv, 0) {
		switch {
		case bits == 32 && math.IsInf(fv, +1):
			return +math.MaxFloat32, true
		case bits == 64 && math.IsInf(fv, +1):
			return +math.MaxFloat64, true
		case bits == 32 && math.IsInf(fv, -1):
			return -math.MaxFloat32, true
		case bits == 64 && math.IsInf(fv, -1):
			return -math.MaxFloat64, true
		}
	}
	return fv, err == nil
}
