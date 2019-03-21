// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// By default, this json support uses base64 encoding for bytes, because you cannot
// store and read any arbitrary string in json (only unicode).
// However, the user can configre how to encode/decode bytes.
//
// This library specifically supports UTF-8 for encoding and decoding only.
//
// Note that the library will happily encode/decode things which are not valid
// json e.g. a map[int64]string. We do it for consistency. With valid json,
// we will encode and decode appropriately.
// Users can specify their map type if necessary to force it.
//
// Note:
//   - we cannot use strconv.Quote and strconv.Unquote because json quotes/unquotes differently.
//     We implement it here.
//   - Also, strconv.ParseXXX for floats and integers
//     - only works on strings resulting in unnecessary allocation and []byte-string conversion.
//     - it does a lot of redundant checks, because json numbers are simpler that what it supports.
//   - We parse numbers (floats and integers) directly here.
//     We only delegate parsing floats if it is a hairy float which could cause a loss of precision.
//     In that case, we delegate to strconv.ParseFloat.
//
// Note:
//   - encode does not beautify. There is no whitespace when encoding.
//   - rpc calls which take single integer arguments or write single numeric arguments will need care.

// Top-level methods of json(End|Dec)Driver (which are implementations of (en|de)cDriver
// MUST not call one-another.

import (
	"bytes"
	"encoding/base64"
	"reflect"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

//--------------------------------

var (
	// jsonLiterals = [...]byte{'t', 'r', 'u', 'e', 'f', 'a', 'l', 's', 'e', 'n', 'u', 'l', 'l'}

	// jsonFloat64Pow10 = [...]float64{
	// 	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	// 	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	// 	1e20, 1e21, 1e22,
	// }

	// jsonUint64Pow10 = [...]uint64{
	// 	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	// 	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	// }

	// jsonTabs and jsonSpaces are used as caches for indents
	jsonTabs, jsonSpaces string

	jsonCharHtmlSafeSet   bitset128
	jsonCharSafeSet       bitset128
	jsonCharWhitespaceSet bitset256
	jsonNumSet            bitset256
	// jsonIsFloatSet        bitset256

	jsonU4Set [256]byte
)

const (
	// jsonUnreadAfterDecNum controls whether we unread after decoding a number.
	//
	// instead of unreading, just update d.tok (iff it's not a whitespace char)
	// However, doing this means that we may HOLD onto some data which belongs to another stream.
	// Thus, it is safest to unread the data when done.
	// keep behind a constant flag for now.
	jsonUnreadAfterDecNum = true

	// If !jsonValidateSymbols, decoding will be faster, by skipping some checks:
	//   - If we see first character of null, false or true,
	//     do not validate subsequent characters.
	//   - e.g. if we see a n, assume null and skip next 3 characters,
	//     and do not validate they are ull.
	// P.S. Do not expect a significant decoding boost from this.
	jsonValidateSymbols = true

	jsonSpacesOrTabsLen = 128

	jsonU4SetErrVal = 128
)

func init() {
	var bs [jsonSpacesOrTabsLen]byte
	for i := 0; i < jsonSpacesOrTabsLen; i++ {
		bs[i] = ' '
	}
	jsonSpaces = string(bs[:])

	for i := 0; i < jsonSpacesOrTabsLen; i++ {
		bs[i] = '\t'
	}
	jsonTabs = string(bs[:])

	// populate the safe values as true: note: ASCII control characters are (0-31)
	// jsonCharSafeSet:     all true except (0-31) " \
	// jsonCharHtmlSafeSet: all true except (0-31) " \ < > &
	var i byte
	for i = 32; i < utf8.RuneSelf; i++ {
		switch i {
		case '"', '\\':
		case '<', '>', '&':
			jsonCharSafeSet.set(i) // = true
		default:
			jsonCharSafeSet.set(i)
			jsonCharHtmlSafeSet.set(i)
		}
	}
	for i = 0; i <= utf8.RuneSelf; i++ {
		switch i {
		case ' ', '\t', '\r', '\n':
			jsonCharWhitespaceSet.set(i)
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'E', '.', '+', '-':
			jsonNumSet.set(i)
		}
	}
	for j := range jsonU4Set {
		switch i = byte(j); i {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			jsonU4Set[i] = i - '0'
		case 'a', 'b', 'c', 'd', 'e', 'f':
			jsonU4Set[i] = i - 'a' + 10
		case 'A', 'B', 'C', 'D', 'E', 'F':
			jsonU4Set[i] = i - 'A' + 10
		default:
			jsonU4Set[i] = jsonU4SetErrVal
		}
		// switch i = byte(j); i {
		// case 'e', 'E', '.':
		// 	jsonIsFloatSet.set(i)
		// }
	}
	// jsonU4Set[255] = jsonU4SetErrVal
}

type jsonEncDriver struct {
	e  *Encoder
	w  encWriter
	h  *JsonHandle
	b  [64]byte // scratch
	bs []byte   // scratch
	se setExtWrapper
	ds string // indent string
	dl uint16 // indent level
	dt bool   // indent using tabs
	d  bool   // indent
	c  containerState
	noBuiltInTypes
}

// indent is done as below:
//   - newline and indent are added before each mapKey or arrayElem
//   - newline and indent are added before each ending,
//     except there was no entry (so we can have {} or [])

func (e *jsonEncDriver) WriteArrayStart(length int) {
	if e.d {
		e.dl++
	}
	e.w.writen1('[')
	e.c = containerArrayStart
}

func (e *jsonEncDriver) WriteArrayElem() {
	if e.c != containerArrayStart {
		e.w.writen1(',')
	}
	if e.d {
		e.writeIndent()
	}
	e.c = containerArrayElem
}

func (e *jsonEncDriver) WriteArrayEnd() {
	if e.d {
		e.dl--
		if e.c != containerArrayStart {
			e.writeIndent()
		}
	}
	e.w.writen1(']')
	e.c = containerArrayEnd
}

func (e *jsonEncDriver) WriteMapStart(length int) {
	if e.d {
		e.dl++
	}
	e.w.writen1('{')
	e.c = containerMapStart
}

func (e *jsonEncDriver) WriteMapElemKey() {
	if e.c != containerMapStart {
		e.w.writen1(',')
	}
	if e.d {
		e.writeIndent()
	}
	e.c = containerMapKey
}

func (e *jsonEncDriver) WriteMapElemValue() {
	if e.d {
		e.w.writen2(':', ' ')
	} else {
		e.w.writen1(':')
	}
	e.c = containerMapValue
}

func (e *jsonEncDriver) WriteMapEnd() {
	if e.d {
		e.dl--
		if e.c != containerMapStart {
			e.writeIndent()
		}
	}
	e.w.writen1('}')
	e.c = containerMapEnd
}

// func (e *jsonEncDriver) sendContainerState(c containerState) {
// 	// determine whether to output separators
// 	switch c {
// 	case containerMapKey:
// 		if e.c != containerMapStart {
// 			e.w.writen1(',')
// 		}
// 		if e.d {
// 			e.writeIndent()
// 		}
// 	case containerMapValue:
// 		if e.d {
// 			e.w.writen2(':', ' ')
// 		} else {
// 			e.w.writen1(':')
// 		}
// 	case containerMapEnd:
// 		if e.d {
// 			e.dl--
// 			if e.c != containerMapStart {
// 				e.writeIndent()
// 			}
// 		}
// 		e.w.writen1('}')
// 	case containerArrayElem:
// 		if e.c != containerArrayStart {
// 			e.w.writen1(',')
// 		}
// 		if e.d {
// 			e.writeIndent()
// 		}
// 	case containerArrayEnd:
// 		if e.d {
// 			e.dl--
// 			if e.c != containerArrayStart {
// 				e.writeIndent()
// 			}
// 		}
// 		e.w.writen1(']')
// 	}
// 	e.c = c
// }

func (e *jsonEncDriver) writeIndent() {
	e.w.writen1('\n')
	if x := len(e.ds) * int(e.dl); x <= jsonSpacesOrTabsLen {
		if e.dt {
			e.w.writestr(jsonTabs[:x])
		} else {
			e.w.writestr(jsonSpaces[:x])
		}
	} else {
		for i := uint16(0); i < e.dl; i++ {
			e.w.writestr(e.ds)
		}
	}
}

func (e *jsonEncDriver) EncodeNil() {
	e.w.writen4('n', 'u', 'l', 'l') // e.w.writeb(jsonLiterals[9:13]) // null
}

func (e *jsonEncDriver) EncodeBool(b bool) {
	if b {
		e.w.writen4('t', 'r', 'u', 'e') // e.w.writeb(jsonLiterals[0:4]) // true
	} else {
		e.w.writen5('f', 'a', 'l', 's', 'e') // e.w.writeb(jsonLiterals[4:9]) // false
	}
}

func (e *jsonEncDriver) EncodeFloat32(f float32) {
	e.encodeFloat(float64(f), 32)
}

func (e *jsonEncDriver) EncodeFloat64(f float64) {
	e.encodeFloat(f, 64)
}

func (e *jsonEncDriver) encodeFloat(f float64, numbits int) {
	x := strconv.AppendFloat(e.b[:0], f, 'G', -1, numbits)
	// if bytes.IndexByte(x, 'E') == -1 && bytes.IndexByte(x, '.') == -1 {
	if !jsonIsFloatBytesB2(x) {
		x = append(x, '.', '0')
	}
	e.w.writeb(x)
}

func (e *jsonEncDriver) EncodeInt(v int64) {
	if x := e.h.IntegerAsString; x == 'A' || x == 'L' && (v > 1<<53 || v < -(1<<53)) {
		e.w.writen1('"')
		e.w.writeb(strconv.AppendInt(e.b[:0], v, 10))
		e.w.writen1('"')
		return
	}
	e.w.writeb(strconv.AppendInt(e.b[:0], v, 10))
}

func (e *jsonEncDriver) EncodeUint(v uint64) {
	if x := e.h.IntegerAsString; x == 'A' || x == 'L' && v > 1<<53 {
		e.w.writen1('"')
		e.w.writeb(strconv.AppendUint(e.b[:0], v, 10))
		e.w.writen1('"')
		return
	}
	e.w.writeb(strconv.AppendUint(e.b[:0], v, 10))
}

func (e *jsonEncDriver) EncodeExt(rv interface{}, xtag uint64, ext Ext, en *Encoder) {
	if v := ext.ConvertExt(rv); v == nil {
		e.w.writen4('n', 'u', 'l', 'l') // e.w.writeb(jsonLiterals[9:13]) // null // e.EncodeNil()
	} else {
		en.encode(v)
	}
}

func (e *jsonEncDriver) EncodeRawExt(re *RawExt, en *Encoder) {
	// only encodes re.Value (never re.Data)
	if re.Value == nil {
		e.w.writen4('n', 'u', 'l', 'l') // e.w.writeb(jsonLiterals[9:13]) // null // e.EncodeNil()
	} else {
		en.encode(re.Value)
	}
}

func (e *jsonEncDriver) EncodeString(c charEncoding, v string) {
	e.quoteStr(v)
}

func (e *jsonEncDriver) EncodeSymbol(v string) {
	e.quoteStr(v)
}

func (e *jsonEncDriver) EncodeStringBytes(c charEncoding, v []byte) {
	// if encoding raw bytes and RawBytesExt is configured, use it to encode
	if c == c_RAW && e.se.i != nil {
		e.EncodeExt(v, 0, &e.se, e.e)
		return
	}
	if c == c_RAW {
		slen := base64.StdEncoding.EncodedLen(len(v))
		if cap(e.bs) >= slen {
			e.bs = e.bs[:slen]
		} else {
			e.bs = make([]byte, slen)
		}
		base64.StdEncoding.Encode(e.bs, v)
		e.w.writen1('"')
		e.w.writeb(e.bs)
		e.w.writen1('"')
	} else {
		e.quoteStr(stringView(v))
	}
}

func (e *jsonEncDriver) EncodeAsis(v []byte) {
	e.w.writeb(v)
}

func (e *jsonEncDriver) quoteStr(s string) {
	// adapted from std pkg encoding/json
	const hex = "0123456789abcdef"
	w := e.w
	w.writen1('"')
	var start int
	for i, slen := 0, len(s); i < slen; {
		// encode all bytes < 0x20 (except \r, \n).
		// also encode < > & to prevent security holes when served to some browsers.
		if b := s[i]; b < utf8.RuneSelf {
			// if 0x20 <= b && b != '\\' && b != '"' && b != '<' && b != '>' && b != '&' {
			if jsonCharHtmlSafeSet.isset(b) || (e.h.HTMLCharsAsIs && jsonCharSafeSet.isset(b)) {
				i++
				continue
			}
			if start < i {
				w.writestr(s[start:i])
			}
			switch b {
			case '\\', '"':
				w.writen2('\\', b)
			case '\n':
				w.writen2('\\', 'n')
			case '\r':
				w.writen2('\\', 'r')
			case '\b':
				w.writen2('\\', 'b')
			case '\f':
				w.writen2('\\', 'f')
			case '\t':
				w.writen2('\\', 't')
			default:
				w.writestr(`\u00`)
				w.writen2(hex[b>>4], hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				w.writestr(s[start:i])
			}
			w.writestr(`\ufffd`)
			i += size
			start = i
			continue
		}
		// U+2028 is LINE SEPARATOR. U+2029 is PARAGRAPH SEPARATOR.
		// Both technically valid JSON, but bomb on JSONP, so fix here unconditionally.
		if c == '\u2028' || c == '\u2029' {
			if start < i {
				w.writestr(s[start:i])
			}
			w.writestr(`\u202`)
			w.writen1(hex[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		w.writestr(s[start:])
	}
	w.writen1('"')
}

func (e *jsonEncDriver) atEndOfEncode() {
	if e.h.TermWhitespace {
		if e.d {
			e.w.writen1('\n')
		} else {
			e.w.writen1(' ')
		}
	}
}

type jsonDecDriver struct {
	noBuiltInTypes
	d *Decoder
	h *JsonHandle
	r decReader

	c containerState
	// tok is used to store the token read right after skipWhiteSpace.
	tok uint8

	fnull bool // found null from appendStringAsBytes

	bstr [8]byte  // scratch used for string \UXXX parsing
	b    [64]byte // scratch, used for parsing strings or numbers
	b2   [64]byte // scratch, used only for decodeBytes (after base64)
	bs   []byte   // scratch. Initialized from b. Used for parsing strings or numbers.

	se setExtWrapper

	// n jsonNum
}

func jsonIsWS(b byte) bool {
	// return b == ' ' || b == '\t' || b == '\r' || b == '\n'
	return jsonCharWhitespaceSet.isset(b)
}

func (d *jsonDecDriver) uncacheRead() {
	if d.tok != 0 {
		d.r.unreadn1()
		d.tok = 0
	}
}

func (d *jsonDecDriver) ReadMapStart() int {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.tok != '{' {
		d.d.errorf("json: expect char '%c' but got char '%c'", '{', d.tok)
	}
	d.tok = 0
	d.c = containerMapStart
	return -1
}

func (d *jsonDecDriver) ReadArrayStart() int {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.tok != '[' {
		d.d.errorf("json: expect char '%c' but got char '%c'", '[', d.tok)
	}
	d.tok = 0
	d.c = containerArrayStart
	return -1
}

func (d *jsonDecDriver) CheckBreak() bool {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	return d.tok == '}' || d.tok == ']'
}

func (d *jsonDecDriver) ReadArrayElem() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.c != containerArrayStart {
		const xc uint8 = ','
		if d.tok != xc {
			d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
		}
		d.tok = 0
	}
	d.c = containerArrayElem
}

func (d *jsonDecDriver) ReadArrayEnd() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	const xc uint8 = ']'
	if d.tok != xc {
		d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
	}
	d.tok = 0
	d.c = containerArrayEnd
}

func (d *jsonDecDriver) ReadMapElemKey() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.c != containerMapStart {
		const xc uint8 = ','
		if d.tok != xc {
			d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
		}
		d.tok = 0
	}
	d.c = containerMapKey
}

func (d *jsonDecDriver) ReadMapElemValue() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	const xc uint8 = ':'
	if d.tok != xc {
		d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
	}
	d.tok = 0
	d.c = containerMapValue
}

func (d *jsonDecDriver) ReadMapEnd() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	const xc uint8 = '}'
	if d.tok != xc {
		d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
	}
	d.tok = 0
	d.c = containerMapEnd
}

// func (d *jsonDecDriver) readContainerState(c containerState, xc uint8, check bool) {
// 	if d.tok == 0 {
// 		d.tok = d.r.skip(&jsonCharWhitespaceSet)
// 	}
// 	if check {
// 		if d.tok != xc {
// 			d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
// 		}
// 		d.tok = 0
// 	}
// 	d.c = c
// }

// func (d *jsonDecDriver) sendContainerState(c containerState) {
// 	if d.tok == 0 {
// 		d.tok = d.r.skip(&jsonCharWhitespaceSet)
// 	}
// 	var xc uint8 // char expected
// 	switch c {
// 	case containerMapKey:
// 		if d.c != containerMapStart {
// 			xc = ','
// 		}
// 	case containerMapValue:
// 		xc = ':'
// 	case containerMapEnd:
// 		xc = '}'
// 	case containerArrayElem:
// 		if d.c != containerArrayStart {
// 			xc = ','
// 		}
// 	case containerArrayEnd:
// 		xc = ']'
// 	}
// 	if xc != 0 {
// 		if d.tok != xc {
// 			d.d.errorf("json: expect char '%c' but got char '%c'", xc, d.tok)
// 		}
// 		d.tok = 0
// 	}
// 	d.c = c
// }

// func (d *jsonDecDriver) readLiteralIdx(fromIdx, toIdx uint8) {
// 	bs := d.r.readx(int(toIdx - fromIdx))
// 	d.tok = 0
// 	if jsonValidateSymbols && !bytes.Equal(bs, jsonLiterals[fromIdx:toIdx]) {
// 		d.d.errorf("json: expecting %s: got %s", jsonLiterals[fromIdx:toIdx], bs)
// 		return
// 	}
// }

func (d *jsonDecDriver) readSymbol3(v1, v2, v3 uint8) {
	b1, b2, b3 := d.r.readn3()
	d.tok = 0
	if jsonValidateSymbols && (b1 != v1 || b2 != v2 || b3 != v3) {
		d.d.errorf("json: expecting %c, %c, %c: got %c, %c, %c", b1, b2, b3, v1, v2, v3)
		return
	}
}

func (d *jsonDecDriver) readSymbol4(v1, v2, v3, v4 uint8) {
	b1, b2, b3, b4 := d.r.readn4()
	d.tok = 0
	if jsonValidateSymbols && (b1 != v1 || b2 != v2 || b3 != v3 || b4 != v4) {
		d.d.errorf("json: expecting %c, %c, %c, %c: got %c, %c, %c, %c", b1, b2, b3, b4, v1, v2, v3, v4)
		return
	}
}

func (d *jsonDecDriver) TryDecodeAsNil() bool {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.tok == 'n' {
		d.readSymbol3('u', 'l', 'l') // d.readLiteralIdx(10, 13) // ull
		return true
	}
	return false
}

func (d *jsonDecDriver) DecodeBool() bool {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.tok == 'f' {
		d.readSymbol4('a', 'l', 's', 'e') // d.readLiteralIdx(5, 9) // alse
		return false
	}
	if d.tok == 't' {
		d.readSymbol3('r', 'u', 'e') // d.readLiteralIdx(1, 4) // rue
		return true
	}
	d.d.errorf("json: decode bool: got first char %c", d.tok)
	return false // "unreachable"
}

func (d *jsonDecDriver) ContainerType() (vt valueType) {
	// check container type by checking the first char
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if b := d.tok; b == '{' {
		return valueTypeMap
	} else if b == '[' {
		return valueTypeArray
	} else if b == 'n' {
		return valueTypeNil
	} else if b == '"' {
		return valueTypeString
	}
	return valueTypeUnset
	// d.d.errorf("isContainerType: unsupported parameter: %v", vt)
	// return false // "unreachable"
}

func (d *jsonDecDriver) decNumBytes() (bs []byte) {
	// stores num bytes in d.bs
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	if d.tok == '"' {
		bs = d.r.readUntil(d.b2[:0], '"')
		bs = bs[:len(bs)-1]
	} else {
		d.r.unreadn1()
		bs = d.r.readTo(d.bs[:0], &jsonNumSet)
	}
	d.tok = 0
	return bs
}

func (d *jsonDecDriver) DecodeUint(bitsize uint8) (u uint64) {
	bs := d.decNumBytes()
	u, err := strconv.ParseUint(stringView(bs), 10, int(bitsize))
	if err != nil {
		d.d.errorf("json: decode uint from %s: %v", bs, err)
		return
	}
	return
}

func (d *jsonDecDriver) DecodeInt(bitsize uint8) (i int64) {
	bs := d.decNumBytes()
	i, err := strconv.ParseInt(stringView(bs), 10, int(bitsize))
	if err != nil {
		d.d.errorf("json: decode int from %s: %v", bs, err)
		return
	}
	return
}

func (d *jsonDecDriver) DecodeFloat(chkOverflow32 bool) (f float64) {
	bs := d.decNumBytes()
	bitsize := 64
	if chkOverflow32 {
		bitsize = 32
	}
	f, err := strconv.ParseFloat(stringView(bs), bitsize)
	if err != nil {
		d.d.errorf("json: decode float from %s: %v", bs, err)
		return
	}
	return
}

func (d *jsonDecDriver) DecodeExt(rv interface{}, xtag uint64, ext Ext) (realxtag uint64) {
	if ext == nil {
		re := rv.(*RawExt)
		re.Tag = xtag
		d.d.decode(&re.Value)
	} else {
		var v interface{}
		d.d.decode(&v)
		ext.UpdateExt(rv, v)
	}
	return
}

func (d *jsonDecDriver) DecodeBytes(bs []byte, zerocopy bool) (bsOut []byte) {
	// if decoding into raw bytes, and the RawBytesExt is configured, use it to decode.
	if d.se.i != nil {
		bsOut = bs
		d.DecodeExt(&bsOut, 0, &d.se)
		return
	}
	d.appendStringAsBytes()
	// base64 encodes []byte{} as "", and we encode nil []byte as null.
	// Consequently, base64 should decode null as a nil []byte, and "" as an empty []byte{}.
	// appendStringAsBytes returns a zero-len slice for both, so as not to reset d.bs.
	// However, it sets a fnull field to true, so we can check if a null was found.
	if len(d.bs) == 0 {
		if d.fnull {
			return nil
		}
		return []byte{}
	}
	bs0 := d.bs
	slen := base64.StdEncoding.DecodedLen(len(bs0))
	if slen <= cap(bs) {
		bsOut = bs[:slen]
	} else if zerocopy && slen <= cap(d.b2) {
		bsOut = d.b2[:slen]
	} else {
		bsOut = make([]byte, slen)
	}
	slen2, err := base64.StdEncoding.Decode(bsOut, bs0)
	if err != nil {
		d.d.errorf("json: error decoding base64 binary '%s': %v", bs0, err)
		return nil
	}
	if slen != slen2 {
		bsOut = bsOut[:slen2]
	}
	return
}

const jsonAlwaysReturnInternString = false

func (d *jsonDecDriver) DecodeString() (s string) {
	d.appendStringAsBytes()
	// if x := d.s.sc; x != nil && x.so && x.st == '}' { // map key
	if jsonAlwaysReturnInternString || d.c == containerMapKey {
		return d.d.string(d.bs)
	}
	return string(d.bs)
}

func (d *jsonDecDriver) DecodeStringAsBytes() (s []byte) {
	d.appendStringAsBytes()
	return d.bs
}

func (d *jsonDecDriver) appendStringAsBytes() {
	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}

	d.fnull = false
	if d.tok != '"' {
		// d.d.errorf("json: expect char '%c' but got char '%c'", '"', d.tok)
		// handle non-string scalar: null, true, false or a number
		switch d.tok {
		case 'n':
			d.readSymbol3('u', 'l', 'l') // d.readLiteralIdx(10, 13) // ull
			d.bs = d.bs[:0]
			d.fnull = true
		case 'f':
			d.readSymbol4('a', 'l', 's', 'e') // d.readLiteralIdx(5, 9) // alse
			d.bs = d.bs[:5]
			copy(d.bs, "false")
		case 't':
			d.readSymbol3('r', 'u', 'e') // d.readLiteralIdx(1, 4) // rue
			d.bs = d.bs[:4]
			copy(d.bs, "true")
		default:
			// try to parse a valid number
			bs := d.decNumBytes()
			d.bs = d.bs[:len(bs)]
			copy(d.bs, bs)
		}
		return
	}

	d.tok = 0
	r := d.r
	var cs = r.readUntil(d.b2[:0], '"')
	var cslen = len(cs)
	var c uint8
	v := d.bs[:0]
	// append on each byte seen can be expensive, so we just
	// keep track of where we last read a contiguous set of
	// non-special bytes (using cursor variable),
	// and when we see a special byte
	// e.g. end-of-slice, " or \,
	// we will append the full range into the v slice before proceeding
	for i, cursor := 0, 0; ; {
		if i == cslen {
			v = append(v, cs[cursor:]...)
			cs = r.readUntil(d.b2[:0], '"')
			cslen = len(cs)
			i, cursor = 0, 0
		}
		c = cs[i]
		if c == '"' {
			v = append(v, cs[cursor:i]...)
			break
		}
		if c != '\\' {
			i++
			continue
		}
		v = append(v, cs[cursor:i]...)
		i++
		c = cs[i]
		switch c {
		case '"', '\\', '/', '\'':
			v = append(v, c)
		case 'b':
			v = append(v, '\b')
		case 'f':
			v = append(v, '\f')
		case 'n':
			v = append(v, '\n')
		case 'r':
			v = append(v, '\r')
		case 't':
			v = append(v, '\t')
		case 'u':
			var r rune
			var rr uint32
			c = cs[i+4] // may help reduce bounds-checking
			for j := 1; j < 5; j++ {
				c = jsonU4Set[cs[i+j]]
				if c == jsonU4SetErrVal {
					d.d.errorf(`json: unquoteStr: invalid hex char in \u unicode sequence: %q`, c)
				}
				rr = rr*16 + uint32(c)
			}
			r = rune(rr)
			i += 4
			if utf16.IsSurrogate(r) {
				if !(cs[i+2] == 'u' && cs[i+i] == '\\') {
					d.d.errorf(`json: unquoteStr: invalid unicode sequence. Expecting \u`)
					return
				}
				i += 2
				c = cs[i+4] // may help reduce bounds-checking
				var rr1 uint32
				for j := 1; j < 5; j++ {
					c = jsonU4Set[cs[i+j]]
					if c == jsonU4SetErrVal {
						d.d.errorf(`json: unquoteStr: invalid hex char in \u unicode sequence: %q`, c)
					}
					rr1 = rr1*16 + uint32(c)
				}
				r = utf16.DecodeRune(r, rune(rr1))
				i += 4
			}
			w2 := utf8.EncodeRune(d.bstr[:], r)
			v = append(v, d.bstr[:w2]...)
		default:
			d.d.errorf("json: unsupported escaped value: %c", c)
		}
		i++
		cursor = i
	}
	d.bs = v
}

// func (d *jsonDecDriver) jsonU4Arr(bs [4]byte) (r rune) {
// 	// u, _ := strconv.ParseUint(string(d.bstr[:4]), 16, 64)
// 	var u uint32
// 	for _, v := range bs {
// 		if '0' <= v && v <= '9' {
// 			v = v - '0'
// 		} else if 'a' <= v && v <= 'f' {
// 			v = v - 'a' + 10
// 		} else if 'A' <= v && v <= 'f' {
// 			v = v - 'A' + 10
// 		} else {
// 			// d.d.errorf(`json: unquoteStr: invalid hex char in \u unicode sequence: %q`, v)
// 			return utf8.RuneError
// 		}
// 		u = u*16 + uint32(v)
// 	}
// 	return rune(u)
// }

func (d *jsonDecDriver) DecodeNaked() {
	z := d.d.n
	// var decodeFurther bool

	if d.tok == 0 {
		d.tok = d.r.skip(&jsonCharWhitespaceSet)
	}
	switch d.tok {
	case 'n':
		d.readSymbol3('u', 'l', 'l') // d.readLiteralIdx(10, 13) // ull
		z.v = valueTypeNil
	case 'f':
		d.readSymbol4('a', 'l', 's', 'e') // d.readLiteralIdx(5, 9) // alse
		z.v = valueTypeBool
		z.b = false
	case 't':
		d.readSymbol3('r', 'u', 'e') // d.readLiteralIdx(1, 4) // rue
		z.v = valueTypeBool
		z.b = true
	case '{':
		z.v = valueTypeMap // don't consume. kInterfaceNaked will call ReadMapStart
	case '[':
		z.v = valueTypeArray // don't consume. kInterfaceNaked will call ReadArrayStart
	case '"':
		z.v = valueTypeString
		z.s = d.DecodeString()
	default: // number
		bs := d.decNumBytes()
		var err error
		if len(bs) == 0 {
			d.d.errorf("json: decode number from empty string")
			return
		} else if d.h.PreferFloat || jsonIsFloatBytesB3(bs) { // bytes.IndexByte(bs, '.') != -1 ||...
			// } else if d.h.PreferFloat || bytes.ContainsAny(bs, ".eE") {
			z.v = valueTypeFloat
			z.f, err = strconv.ParseFloat(stringView(bs), 64)
		} else if d.h.SignedInteger || bs[0] == '-' {
			z.v = valueTypeInt
			z.i, err = strconv.ParseInt(stringView(bs), 10, 64)
		} else {
			z.v = valueTypeUint
			z.u, err = strconv.ParseUint(stringView(bs), 10, 64)
		}
		if err != nil {
			if z.v == valueTypeInt || z.v == valueTypeUint {
				if v, ok := err.(*strconv.NumError); ok && (v.Err == strconv.ErrRange || v.Err == strconv.ErrSyntax) {
					z.v = valueTypeFloat
					z.f, err = strconv.ParseFloat(stringView(bs), 64)
				}
			}
			if err != nil {
				d.d.errorf("json: decode number from %s: %v", bs, err)
				return
			}
		}
	}
	// if decodeFurther {
	// 	d.s.sc.retryRead()
	// }
	return
}

//----------------------

// JsonHandle is a handle for JSON encoding format.
//
// Json is comprehensively supported:
//    - decodes numbers into interface{} as int, uint or float64
//    - configurable way to encode/decode []byte .
//      by default, encodes and decodes []byte using base64 Std Encoding
//    - UTF-8 support for encoding and decoding
//
// It has better performance than the json library in the standard library,
// by leveraging the performance improvements of the codec library and
// minimizing allocations.
//
// In addition, it doesn't read more bytes than necessary during a decode, which allows
// reading multiple values from a stream containing json and non-json content.
// For example, a user can read a json value, then a cbor value, then a msgpack value,
// all from the same stream in sequence.
type JsonHandle struct {
	textEncodingType
	BasicHandle

	// RawBytesExt, if configured, is used to encode and decode raw bytes in a custom way.
	// If not configured, raw bytes are encoded to/from base64 text.
	RawBytesExt InterfaceExt

	// Indent indicates how a value is encoded.
	//   - If positive, indent by that number of spaces.
	//   - If negative, indent by that number of tabs.
	Indent int8

	// IntegerAsString controls how integers (signed and unsigned) are encoded.
	//
	// Per the JSON Spec, JSON numbers are 64-bit floating point numbers.
	// Consequently, integers > 2^53 cannot be represented as a JSON number without losing precision.
	// This can be mitigated by configuring how to encode integers.
	//
	// IntegerAsString interpretes the following values:
	//   - if 'L', then encode integers > 2^53 as a json string.
	//   - if 'A', then encode all integers as a json string
	//             containing the exact integer representation as a decimal.
	//   - else    encode all integers as a json number (default)
	IntegerAsString uint8

	// HTMLCharsAsIs controls how to encode some special characters to html: < > &
	//
	// By default, we encode them as \uXXX
	// to prevent security holes when served from some browsers.
	HTMLCharsAsIs bool

	// PreferFloat says that we will default to decoding a number as a float.
	// If not set, we will examine the characters of the number and decode as an
	// integer type if it doesn't have any of the characters [.eE].
	PreferFloat bool

	// TermWhitespace says that we add a whitespace character
	// at the end of an encoding.
	//
	// The whitespace is important, especially if using numbers in a context
	// where multiple items are written to a stream.
	TermWhitespace bool
}

func (h *JsonHandle) hasElemSeparators() bool { return true }

func (h *JsonHandle) SetInterfaceExt(rt reflect.Type, tag uint64, ext InterfaceExt) (err error) {
	return h.SetExt(rt, tag, &setExtWrapper{i: ext})
}

func (h *JsonHandle) newEncDriver(e *Encoder) encDriver {
	hd := jsonEncDriver{e: e, h: h}
	hd.bs = hd.b[:0]

	hd.reset()

	return &hd
}

func (h *JsonHandle) newDecDriver(d *Decoder) decDriver {
	// d := jsonDecDriver{r: r.(*bytesDecReader), h: h}
	hd := jsonDecDriver{d: d, h: h}
	hd.bs = hd.b[:0]
	hd.reset()
	return &hd
}

func (e *jsonEncDriver) reset() {
	e.w = e.e.w
	e.se.i = e.h.RawBytesExt
	if e.bs != nil {
		e.bs = e.bs[:0]
	}
	e.d, e.dt, e.dl, e.ds = false, false, 0, ""
	e.c = 0
	if e.h.Indent > 0 {
		e.d = true
		e.ds = jsonSpaces[:e.h.Indent]
	} else if e.h.Indent < 0 {
		e.d = true
		e.dt = true
		e.ds = jsonTabs[:-(e.h.Indent)]
	}
}

func (d *jsonDecDriver) reset() {
	d.r = d.d.r
	d.se.i = d.h.RawBytesExt
	if d.bs != nil {
		d.bs = d.bs[:0]
	}
	d.c, d.tok = 0, 0
	// d.n.reset()
}

// func jsonIsFloatBytes(bs []byte) bool {
// 	for _, v := range bs {
// 		// if v == '.' || v == 'e' || v == 'E' {
// 		if jsonIsFloatSet.isset(v) {
// 			return true
// 		}
// 	}
// 	return false
// }

func jsonIsFloatBytesB2(bs []byte) bool {
	return bytes.IndexByte(bs, '.') != -1 ||
		bytes.IndexByte(bs, 'E') != -1
}

func jsonIsFloatBytesB3(bs []byte) bool {
	return bytes.IndexByte(bs, '.') != -1 ||
		bytes.IndexByte(bs, 'E') != -1 ||
		bytes.IndexByte(bs, 'e') != -1
}

// var jsonEncodeTerminate = []byte{' '}

// func (h *JsonHandle) rpcEncodeTerminate() []byte {
// 	return jsonEncodeTerminate
// }

var _ decDriver = (*jsonDecDriver)(nil)
var _ encDriver = (*jsonEncDriver)(nil)
