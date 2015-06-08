// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// This json support uses base64 encoding for bytes, because you cannot
// store and read any arbitrary string in json (only unicode).
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

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

//--------------------------------

var jsonLiterals = [...]byte{'t', 'r', 'u', 'e', 'f', 'a', 'l', 's', 'e', 'n', 'u', 'l', 'l'}

var jsonFloat64Pow10 = [...]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	1e20, 1e21, 1e22,
}

var jsonUint64Pow10 = [...]uint64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
}

const (
	// if jsonTrackSkipWhitespace, we track Whitespace and reduce the number of redundant checks.
	// Make it a const flag, so that it can be elided during linking if false.
	//
	// It is not a clear win, because we continually set a flag behind a pointer
	// and then check it each time, as opposed to just 4 conditionals on a stack variable.
	jsonTrackSkipWhitespace = true

	// If !jsonValidateSymbols, decoding will be faster, by skipping some checks:
	//   - If we see first character of null, false or true,
	//     do not validate subsequent characters.
	//   - e.g. if we see a n, assume null and skip next 3 characters,
	//     and do not validate they are ull.
	// P.S. Do not expect a significant decoding boost from this.
	jsonValidateSymbols = true

	// if jsonTruncateMantissa, truncate mantissa if trailing 0's.
	// This is important because it could allow some floats to be decoded without
	// deferring to strconv.ParseFloat.
	jsonTruncateMantissa = true

	// if mantissa >= jsonNumUintCutoff before multiplying by 10, this is an overflow
	jsonNumUintCutoff = (1<<64-1)/uint64(10) + 1 // cutoff64(base)

	// if mantissa >= jsonNumUintMaxVal, this is an overflow
	jsonNumUintMaxVal = 1<<uint64(64) - 1

	// jsonNumDigitsUint64Largest = 19
)

type jsonEncDriver struct {
	e  *Encoder
	w  encWriter
	h  *JsonHandle
	b  [64]byte // scratch
	bs []byte   // scratch
	noBuiltInTypes
}

func (e *jsonEncDriver) EncodeNil() {
	e.w.writeb(jsonLiterals[9:13]) // null
}

func (e *jsonEncDriver) EncodeBool(b bool) {
	if b {
		e.w.writeb(jsonLiterals[0:4]) // true
	} else {
		e.w.writeb(jsonLiterals[4:9]) // false
	}
}

func (e *jsonEncDriver) EncodeFloat32(f float32) {
	e.w.writeb(strconv.AppendFloat(e.b[:0], float64(f), 'E', -1, 32))
}

func (e *jsonEncDriver) EncodeFloat64(f float64) {
	// e.w.writestr(strconv.FormatFloat(f, 'E', -1, 64))
	e.w.writeb(strconv.AppendFloat(e.b[:0], f, 'E', -1, 64))
}

func (e *jsonEncDriver) EncodeInt(v int64) {
	e.w.writeb(strconv.AppendInt(e.b[:0], v, 10))
}

func (e *jsonEncDriver) EncodeUint(v uint64) {
	e.w.writeb(strconv.AppendUint(e.b[:0], v, 10))
}

func (e *jsonEncDriver) EncodeExt(rv interface{}, xtag uint64, ext Ext, en *Encoder) {
	if v := ext.ConvertExt(rv); v == nil {
		e.EncodeNil()
	} else {
		en.encode(v)
	}
}

func (e *jsonEncDriver) EncodeRawExt(re *RawExt, en *Encoder) {
	// only encodes re.Value (never re.Data)
	if re.Value == nil {
		e.EncodeNil()
	} else {
		en.encode(re.Value)
	}
}

func (e *jsonEncDriver) EncodeArrayStart(length int) {
	e.w.writen1('[')
}

func (e *jsonEncDriver) EncodeArrayEntrySeparator() {
	e.w.writen1(',')
}

func (e *jsonEncDriver) EncodeArrayEnd() {
	e.w.writen1(']')
}

func (e *jsonEncDriver) EncodeMapStart(length int) {
	e.w.writen1('{')
}

func (e *jsonEncDriver) EncodeMapEntrySeparator() {
	e.w.writen1(',')
}

func (e *jsonEncDriver) EncodeMapKVSeparator() {
	e.w.writen1(':')
}

func (e *jsonEncDriver) EncodeMapEnd() {
	e.w.writen1('}')
}

func (e *jsonEncDriver) EncodeString(c charEncoding, v string) {
	// e.w.writestr(strconv.Quote(v))
	e.quoteStr(v)
}

func (e *jsonEncDriver) EncodeSymbol(v string) {
	// e.EncodeString(c_UTF8, v)
	e.quoteStr(v)
}

func (e *jsonEncDriver) EncodeStringBytes(c charEncoding, v []byte) {
	if c == c_RAW {
		slen := base64.StdEncoding.EncodedLen(len(v))
		if e.bs == nil {
			e.bs = e.b[:]
		}
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
		// e.EncodeString(c, string(v))
		e.quoteStr(stringView(v))
	}
}

func (e *jsonEncDriver) quoteStr(s string) {
	// adapted from std pkg encoding/json
	const hex = "0123456789abcdef"
	w := e.w
	w.writen1('"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if 0x20 <= b && b != '\\' && b != '"' && b != '<' && b != '>' && b != '&' {
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
				// encode all bytes < 0x20 (except \r, \n).
				// also encode < > & to prevent security holes when served to some browsers.
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
		// Both technically valid JSON, but bomb on JSONP, so fix here.
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

//--------------------------------

type jsonNum struct {
	bytes            []byte // may have [+-.eE0-9]
	mantissa         uint64 // where mantissa ends, and maybe dot begins.
	exponent         int16  // exponent value.
	manOverflow      bool
	neg              bool // started with -. No initial sign in the bytes above.
	dot              bool // has dot
	explicitExponent bool // explicit exponent
}

func (x *jsonNum) reset() {
	x.bytes = x.bytes[:0]
	x.manOverflow = false
	x.neg = false
	x.dot = false
	x.explicitExponent = false
	x.mantissa = 0
	x.exponent = 0
}

// uintExp is called only if exponent > 0.
func (x *jsonNum) uintExp() (n uint64, overflow bool) {
	n = x.mantissa
	e := x.exponent
	if e >= int16(len(jsonUint64Pow10)) {
		overflow = true
		return
	}
	n *= jsonUint64Pow10[e]
	if n < x.mantissa || n > jsonNumUintMaxVal {
		overflow = true
		return
	}
	return
	// for i := int16(0); i < e; i++ {
	// 	if n >= jsonNumUintCutoff {
	// 		overflow = true
	// 		return
	// 	}
	// 	n *= 10
	// }
	// return
}

func (x *jsonNum) floatVal() (f float64) {
	// We do not want to lose precision.
	// Consequently, we will delegate to strconv.ParseFloat if any of the following happen:
	//    - There are more digits than in math.MaxUint64: 18446744073709551615 (20 digits)
	//      We expect up to 99.... (19 digits)
	//    - The mantissa cannot fit into a 52 bits of uint64
	//    - The exponent is beyond our scope ie beyong 22.
	const uint64MantissaBits = 52
	const maxExponent = int16(len(jsonFloat64Pow10)) - 1

	parseUsingStrConv := x.manOverflow ||
		x.exponent > maxExponent ||
		(x.exponent < 0 && -(x.exponent) > maxExponent) ||
		x.mantissa>>uint64MantissaBits != 0
	if parseUsingStrConv {
		var err error
		if f, err = strconv.ParseFloat(stringView(x.bytes), 64); err != nil {
			panic(fmt.Errorf("parse float: %s, %v", x.bytes, err))
			return
		}
		if x.neg {
			f = -f
		}
		return
	}

	// all good. so handle parse here.
	f = float64(x.mantissa)
	// fmt.Printf(".Float: uint64 value: %v, float: %v\n", m, f)
	if x.neg {
		f = -f
	}
	if x.exponent > 0 {
		f *= jsonFloat64Pow10[x.exponent]
	} else if x.exponent < 0 {
		f /= jsonFloat64Pow10[-x.exponent]
	}
	return
}

type jsonDecDriver struct {
	d    *Decoder
	h    *JsonHandle
	r    decReader // *bytesDecReader decReader
	ct   valueType // container type. one of unset, array or map.
	bstr [8]byte   // scratch used for string \UXXX parsing
	b    [64]byte  // scratch

	wsSkipped bool // whitespace skipped

	n jsonNum
	noBuiltInTypes
}

// This will skip whitespace characters and return the next byte to read.
// The next byte determines what the value will be one of.
func (d *jsonDecDriver) skipWhitespace(unread bool) (b byte) {
	// as initReadNext is not called all the time, we set ct to unSet whenever
	// we skipwhitespace, as this is the signal that something new is about to be read.
	d.ct = valueTypeUnset
	b = d.r.readn1()
	if !jsonTrackSkipWhitespace || !d.wsSkipped {
		for ; b == ' ' || b == '\t' || b == '\r' || b == '\n'; b = d.r.readn1() {
		}
		if jsonTrackSkipWhitespace {
			d.wsSkipped = true
		}
	}
	if unread {
		d.r.unreadn1()
	}
	return b
}

func (d *jsonDecDriver) CheckBreak() bool {
	b := d.skipWhitespace(true)
	return b == '}' || b == ']'
}

func (d *jsonDecDriver) readStrIdx(fromIdx, toIdx uint8) {
	bs := d.r.readx(int(toIdx - fromIdx))
	if jsonValidateSymbols {
		if !bytes.Equal(bs, jsonLiterals[fromIdx:toIdx]) {
			d.d.errorf("json: expecting %s: got %s", jsonLiterals[fromIdx:toIdx], bs)
			return
		}
	}
	if jsonTrackSkipWhitespace {
		d.wsSkipped = false
	}
}

func (d *jsonDecDriver) TryDecodeAsNil() bool {
	b := d.skipWhitespace(true)
	if b == 'n' {
		d.readStrIdx(9, 13) // null
		d.ct = valueTypeNil
		return true
	}
	return false
}

func (d *jsonDecDriver) DecodeBool() bool {
	b := d.skipWhitespace(false)
	if b == 'f' {
		d.readStrIdx(5, 9) // alse
		return false
	}
	if b == 't' {
		d.readStrIdx(1, 4) // rue
		return true
	}
	d.d.errorf("json: decode bool: got first char %c", b)
	return false // "unreachable"
}

func (d *jsonDecDriver) ReadMapStart() int {
	d.expectChar('{')
	d.ct = valueTypeMap
	return -1
}

func (d *jsonDecDriver) ReadArrayStart() int {
	d.expectChar('[')
	d.ct = valueTypeArray
	return -1
}
func (d *jsonDecDriver) ReadMapEnd() {
	d.expectChar('}')
}
func (d *jsonDecDriver) ReadArrayEnd() {
	d.expectChar(']')
}
func (d *jsonDecDriver) ReadArrayEntrySeparator() {
	d.expectChar(',')
}
func (d *jsonDecDriver) ReadMapEntrySeparator() {
	d.expectChar(',')
}
func (d *jsonDecDriver) ReadMapKVSeparator() {
	d.expectChar(':')
}
func (d *jsonDecDriver) expectChar(c uint8) {
	b := d.skipWhitespace(false)
	if b != c {
		d.d.errorf("json: expect char %c but got char %c", c, b)
		return
	}
	if jsonTrackSkipWhitespace {
		d.wsSkipped = false
	}
}

func (d *jsonDecDriver) IsContainerType(vt valueType) bool {
	// check container type by checking the first char
	if d.ct == valueTypeUnset {
		b := d.skipWhitespace(true)
		if b == '{' {
			d.ct = valueTypeMap
		} else if b == '[' {
			d.ct = valueTypeArray
		} else if b == 'n' {
			d.ct = valueTypeNil
		} else if b == '"' {
			d.ct = valueTypeString
		}
	}
	if vt == valueTypeNil || vt == valueTypeBytes || vt == valueTypeString ||
		vt == valueTypeArray || vt == valueTypeMap {
		return d.ct == vt
	}
	// ugorji: made switch into conditionals, so that IsContainerType can be inlined.
	// switch vt {
	// case valueTypeNil, valueTypeBytes, valueTypeString, valueTypeArray, valueTypeMap:
	// 	return d.ct == vt
	// }
	d.d.errorf("isContainerType: unsupported parameter: %v", vt)
	return false // "unreachable"
}

func (d *jsonDecDriver) decNum(storeBytes bool) {
	// storeBytes = true // TODO: remove.

	// If it is has a . or an e|E, decode as a float; else decode as an int.
	b := d.skipWhitespace(false)
	if !(b == '+' || b == '-' || b == '.' || (b >= '0' && b <= '9')) {
		d.d.errorf("json: decNum: got first char '%c'", b)
		return
	}

	const cutoff = (1<<64-1)/uint64(10) + 1 // cutoff64(base)
	const jsonNumUintMaxVal = 1<<uint64(64) - 1

	// var n jsonNum // create stack-copy jsonNum, and set to pointer at end.
	// n.bytes = d.n.bytes[:0]
	n := &d.n
	n.reset()

	// The format of a number is as below:
	// parsing:     sign? digit* dot? digit* e?  sign? digit*
	// states:  0   1*    2      3*   4      5*  6     7
	// We honor this state so we can break correctly.
	var state uint8 = 0
	var eNeg bool
	var e int16
	var eof bool
LOOP:
	for !eof {
		// fmt.Printf("LOOP: b: %q\n", b)
		switch b {
		case '+':
			switch state {
			case 0:
				state = 2
				// do not add sign to the slice ...
				b, eof = d.r.readn1eof()
				continue
			case 6: // typ = jsonNumFloat
				state = 7
			default:
				break LOOP
			}
		case '-':
			switch state {
			case 0:
				state = 2
				n.neg = true
				// do not add sign to the slice ...
				b, eof = d.r.readn1eof()
				continue
			case 6: // typ = jsonNumFloat
				eNeg = true
				state = 7
			default:
				break LOOP
			}
		case '.':
			switch state {
			case 0, 2: // typ = jsonNumFloat
				state = 4
				n.dot = true
			default:
				break LOOP
			}
		case 'e', 'E':
			switch state {
			case 0, 2, 4: // typ = jsonNumFloat
				state = 6
				// n.mantissaEndIndex = int16(len(n.bytes))
				n.explicitExponent = true
			default:
				break LOOP
			}
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			switch state {
			case 0:
				state = 2
				fallthrough
			case 2:
				fallthrough
			case 4:
				if n.dot {
					n.exponent--
				}
				if n.mantissa >= jsonNumUintCutoff {
					n.manOverflow = true
					break
				}
				v := uint64(b - '0')
				n.mantissa *= 10
				if v != 0 {
					n1 := n.mantissa + v
					if n1 < n.mantissa || n1 > jsonNumUintMaxVal {
						n.manOverflow = true // n+v overflows
						break
					}
					n.mantissa = n1
				}
			case 6:
				state = 7
				fallthrough
			case 7:
				if !(b == '0' && e == 0) {
					e = e*10 + int16(b-'0')
				}
			default:
				break LOOP
			}
		default:
			break LOOP
		}
		if storeBytes {
			n.bytes = append(n.bytes, b)
		}
		b, eof = d.r.readn1eof()
	}

	if jsonTruncateMantissa && n.mantissa != 0 {
		for n.mantissa%10 == 0 {
			n.mantissa /= 10
			n.exponent++
		}
	}

	if e != 0 {
		if eNeg {
			n.exponent -= e
		} else {
			n.exponent += e
		}
	}

	// d.n = n

	if !eof {
		d.r.unreadn1()
	}
	if jsonTrackSkipWhitespace {
		d.wsSkipped = false
	}
	// fmt.Printf("1: n: bytes: %s, neg: %v, dot: %v, exponent: %v, mantissaEndIndex: %v\n",
	// 	n.bytes, n.neg, n.dot, n.exponent, n.mantissaEndIndex)
	return
}

func (d *jsonDecDriver) DecodeInt(bitsize uint8) (i int64) {
	d.decNum(false)
	n := &d.n
	if n.manOverflow {
		d.d.errorf("json: overflow integer after: %v", n.mantissa)
		return
	}
	var u uint64
	if n.exponent == 0 {
		u = n.mantissa
	} else if n.exponent < 0 {
		d.d.errorf("json: fractional integer")
		return
	} else if n.exponent > 0 {
		var overflow bool
		if u, overflow = n.uintExp(); overflow {
			d.d.errorf("json: overflow integer")
			return
		}
	}
	i = int64(u)
	if n.neg {
		i = -i
	}
	if chkOvf.Int(i, bitsize) {
		d.d.errorf("json: overflow %v bits: %s", bitsize, n.bytes)
		return
	}
	// fmt.Printf("DecodeInt: %v\n", i)
	return
}

func (d *jsonDecDriver) DecodeUint(bitsize uint8) (u uint64) {
	d.decNum(false)
	n := &d.n
	if n.neg {
		d.d.errorf("json: unsigned integer cannot be negative")
		return
	}
	if n.manOverflow {
		d.d.errorf("json: overflow integer after: %v", n.mantissa)
		return
	}
	if n.exponent == 0 {
		u = n.mantissa
	} else if n.exponent < 0 {
		d.d.errorf("json: fractional integer")
		return
	} else if n.exponent > 0 {
		var overflow bool
		if u, overflow = n.uintExp(); overflow {
			d.d.errorf("json: overflow integer")
			return
		}
	}
	if chkOvf.Uint(u, bitsize) {
		d.d.errorf("json: overflow %v bits: %s", bitsize, n.bytes)
		return
	}
	// fmt.Printf("DecodeUint: %v\n", u)
	return
}

func (d *jsonDecDriver) DecodeFloat(chkOverflow32 bool) (f float64) {
	d.decNum(true)
	n := &d.n
	f = n.floatVal()
	if chkOverflow32 && chkOvf.Float32(f) {
		d.d.errorf("json: overflow float32: %v, %s", f, n.bytes)
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

func (d *jsonDecDriver) DecodeBytes(bs []byte, isstring, zerocopy bool) (bsOut []byte) {
	// zerocopy doesn't matter for json, as the bytes must be parsed.
	bs0 := d.appendStringAsBytes(d.b[:0])
	if isstring {
		return bs0
	}
	slen := base64.StdEncoding.DecodedLen(len(bs0))
	if cap(bs) >= slen {
		bsOut = bs[:slen]
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

func (d *jsonDecDriver) DecodeString() (s string) {
	return string(d.appendStringAsBytes(d.b[:0]))
}

func (d *jsonDecDriver) appendStringAsBytes(v []byte) []byte {
	d.expectChar('"')
	for {
		c := d.r.readn1()
		if c == '"' {
			break
		} else if c == '\\' {
			c = d.r.readn1()
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
				rr := d.jsonU4(false)
				// fmt.Printf("$$$$$$$$$: is surrogate: %v\n", utf16.IsSurrogate(rr))
				if utf16.IsSurrogate(rr) {
					rr = utf16.DecodeRune(rr, d.jsonU4(true))
				}
				w2 := utf8.EncodeRune(d.bstr[:], rr)
				v = append(v, d.bstr[:w2]...)
			default:
				d.d.errorf("json: unsupported escaped value: %c", c)
				return nil
			}
		} else {
			v = append(v, c)
		}
	}
	if jsonTrackSkipWhitespace {
		d.wsSkipped = false
	}
	return v
}

func (d *jsonDecDriver) jsonU4(checkSlashU bool) rune {
	if checkSlashU && !(d.r.readn1() == '\\' && d.r.readn1() == 'u') {
		d.d.errorf(`json: unquoteStr: invalid unicode sequence. Expecting \u`)
		return 0
	}
	// u, _ := strconv.ParseUint(string(d.bstr[:4]), 16, 64)
	var u uint32
	for i := 0; i < 4; i++ {
		v := d.r.readn1()
		if '0' <= v && v <= '9' {
			v = v - '0'
		} else if 'a' <= v && v <= 'z' {
			v = v - 'a' + 10
		} else if 'A' <= v && v <= 'Z' {
			v = v - 'A' + 10
		} else {
			d.d.errorf(`json: unquoteStr: invalid hex char in \u unicode sequence: %q`, v)
			return 0
		}
		u = u*16 + uint32(v)
	}
	return rune(u)
}

func (d *jsonDecDriver) DecodeNaked() (v interface{}, vt valueType, decodeFurther bool) {
	n := d.skipWhitespace(true)
	switch n {
	case 'n':
		d.readStrIdx(9, 13) // null
		vt = valueTypeNil
	case 'f':
		d.readStrIdx(4, 9) // false
		vt = valueTypeBool
		v = false
	case 't':
		d.readStrIdx(0, 4) // true
		vt = valueTypeBool
		v = true
	case '{':
		vt = valueTypeMap
		decodeFurther = true
	case '[':
		vt = valueTypeArray
		decodeFurther = true
	case '"':
		vt = valueTypeString
		v = d.DecodeString()
	default: // number
		d.decNum(true)
		n := &d.n
		// if the string had a any of [.eE], then decode as float.
		switch {
		case n.explicitExponent, n.dot, n.exponent < 0, n.manOverflow:
			vt = valueTypeFloat
			v = n.floatVal()
		case n.exponent == 0:
			u := n.mantissa
			switch {
			case n.neg:
				vt = valueTypeInt
				v = -int64(u)
			case d.h.SignedInteger:
				vt = valueTypeInt
				v = int64(u)
			default:
				vt = valueTypeUint
				v = u
			}
		default:
			u, overflow := n.uintExp()
			switch {
			case overflow:
				vt = valueTypeFloat
				v = n.floatVal()
			case n.neg:
				vt = valueTypeInt
				v = -int64(u)
			case d.h.SignedInteger:
				vt = valueTypeInt
				v = int64(u)
			default:
				vt = valueTypeUint
				v = u
			}
		}
		// fmt.Printf("DecodeNaked: Number: %T, %v\n", v, v)
	}
	return
}

//----------------------

// JsonHandle is a handle for JSON encoding format.
//
// Json is comprehensively supported:
//    - decodes numbers into interface{} as int, uint or float64
//    - encodes and decodes []byte using base64 Std Encoding
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
	BasicHandle
	textEncodingType
}

func (h *JsonHandle) newEncDriver(e *Encoder) encDriver {
	return &jsonEncDriver{e: e, w: e.w, h: h}
}

func (h *JsonHandle) newDecDriver(d *Decoder) decDriver {
	// d := jsonDecDriver{r: r.(*bytesDecReader), h: h}
	hd := jsonDecDriver{d: d, r: d.r, h: h}
	hd.n.bytes = d.b[:]
	return &hd
}

var jsonEncodeTerminate = []byte{' '}

func (h *JsonHandle) rpcEncodeTerminate() []byte {
	return jsonEncodeTerminate
}

var _ decDriver = (*jsonDecDriver)(nil)
var _ encDriver = (*jsonEncDriver)(nil)
