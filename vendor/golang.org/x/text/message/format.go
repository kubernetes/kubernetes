// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"bytes"
	"strconv"
	"unicode/utf8"

	"golang.org/x/text/internal/format"
)

const (
	ldigits = "0123456789abcdefx"
	udigits = "0123456789ABCDEFX"
)

const (
	signed   = true
	unsigned = false
)

// A formatInfo is the raw formatter used by Printf etc.
// It prints into a buffer that must be set up separately.
type formatInfo struct {
	buf *bytes.Buffer

	format.Parser

	// intbuf is large enough to store %b of an int64 with a sign and
	// avoids padding at the end of the struct on 32 bit architectures.
	intbuf [68]byte
}

func (f *formatInfo) init(buf *bytes.Buffer) {
	f.ClearFlags()
	f.buf = buf
}

// writePadding generates n bytes of padding.
func (f *formatInfo) writePadding(n int) {
	if n <= 0 { // No padding bytes needed.
		return
	}
	f.buf.Grow(n)
	// Decide which byte the padding should be filled with.
	padByte := byte(' ')
	if f.Zero {
		padByte = byte('0')
	}
	// Fill padding with padByte.
	for i := 0; i < n; i++ {
		f.buf.WriteByte(padByte) // TODO: make more efficient.
	}
}

// pad appends b to f.buf, padded on left (!f.minus) or right (f.minus).
func (f *formatInfo) pad(b []byte) {
	if !f.WidthPresent || f.Width == 0 {
		f.buf.Write(b)
		return
	}
	width := f.Width - utf8.RuneCount(b)
	if !f.Minus {
		// left padding
		f.writePadding(width)
		f.buf.Write(b)
	} else {
		// right padding
		f.buf.Write(b)
		f.writePadding(width)
	}
}

// padString appends s to f.buf, padded on left (!f.minus) or right (f.minus).
func (f *formatInfo) padString(s string) {
	if !f.WidthPresent || f.Width == 0 {
		f.buf.WriteString(s)
		return
	}
	width := f.Width - utf8.RuneCountInString(s)
	if !f.Minus {
		// left padding
		f.writePadding(width)
		f.buf.WriteString(s)
	} else {
		// right padding
		f.buf.WriteString(s)
		f.writePadding(width)
	}
}

// fmt_boolean formats a boolean.
func (f *formatInfo) fmt_boolean(v bool) {
	if v {
		f.padString("true")
	} else {
		f.padString("false")
	}
}

// fmt_unicode formats a uint64 as "U+0078" or with f.sharp set as "U+0078 'x'".
func (f *formatInfo) fmt_unicode(u uint64) {
	buf := f.intbuf[0:]

	// With default precision set the maximum needed buf length is 18
	// for formatting -1 with %#U ("U+FFFFFFFFFFFFFFFF") which fits
	// into the already allocated intbuf with a capacity of 68 bytes.
	prec := 4
	if f.PrecPresent && f.Prec > 4 {
		prec = f.Prec
		// Compute space needed for "U+" , number, " '", character, "'".
		width := 2 + prec + 2 + utf8.UTFMax + 1
		if width > len(buf) {
			buf = make([]byte, width)
		}
	}

	// Format into buf, ending at buf[i]. Formatting numbers is easier right-to-left.
	i := len(buf)

	// For %#U we want to add a space and a quoted character at the end of the buffer.
	if f.Sharp && u <= utf8.MaxRune && strconv.IsPrint(rune(u)) {
		i--
		buf[i] = '\''
		i -= utf8.RuneLen(rune(u))
		utf8.EncodeRune(buf[i:], rune(u))
		i--
		buf[i] = '\''
		i--
		buf[i] = ' '
	}
	// Format the Unicode code point u as a hexadecimal number.
	for u >= 16 {
		i--
		buf[i] = udigits[u&0xF]
		prec--
		u >>= 4
	}
	i--
	buf[i] = udigits[u]
	prec--
	// Add zeros in front of the number until requested precision is reached.
	for prec > 0 {
		i--
		buf[i] = '0'
		prec--
	}
	// Add a leading "U+".
	i--
	buf[i] = '+'
	i--
	buf[i] = 'U'

	oldZero := f.Zero
	f.Zero = false
	f.pad(buf[i:])
	f.Zero = oldZero
}

// fmt_integer formats signed and unsigned integers.
func (f *formatInfo) fmt_integer(u uint64, base int, isSigned bool, digits string) {
	negative := isSigned && int64(u) < 0
	if negative {
		u = -u
	}

	buf := f.intbuf[0:]
	// The already allocated f.intbuf with a capacity of 68 bytes
	// is large enough for integer formatting when no precision or width is set.
	if f.WidthPresent || f.PrecPresent {
		// Account 3 extra bytes for possible addition of a sign and "0x".
		width := 3 + f.Width + f.Prec // wid and prec are always positive.
		if width > len(buf) {
			// We're going to need a bigger boat.
			buf = make([]byte, width)
		}
	}

	// Two ways to ask for extra leading zero digits: %.3d or %03d.
	// If both are specified the f.zero flag is ignored and
	// padding with spaces is used instead.
	prec := 0
	if f.PrecPresent {
		prec = f.Prec
		// Precision of 0 and value of 0 means "print nothing" but padding.
		if prec == 0 && u == 0 {
			oldZero := f.Zero
			f.Zero = false
			f.writePadding(f.Width)
			f.Zero = oldZero
			return
		}
	} else if f.Zero && f.WidthPresent {
		prec = f.Width
		if negative || f.Plus || f.Space {
			prec-- // leave room for sign
		}
	}

	// Because printing is easier right-to-left: format u into buf, ending at buf[i].
	// We could make things marginally faster by splitting the 32-bit case out
	// into a separate block but it's not worth the duplication, so u has 64 bits.
	i := len(buf)
	// Use constants for the division and modulo for more efficient code.
	// Switch cases ordered by popularity.
	switch base {
	case 10:
		for u >= 10 {
			i--
			next := u / 10
			buf[i] = byte('0' + u - next*10)
			u = next
		}
	case 16:
		for u >= 16 {
			i--
			buf[i] = digits[u&0xF]
			u >>= 4
		}
	case 8:
		for u >= 8 {
			i--
			buf[i] = byte('0' + u&7)
			u >>= 3
		}
	case 2:
		for u >= 2 {
			i--
			buf[i] = byte('0' + u&1)
			u >>= 1
		}
	default:
		panic("fmt: unknown base; can't happen")
	}
	i--
	buf[i] = digits[u]
	for i > 0 && prec > len(buf)-i {
		i--
		buf[i] = '0'
	}

	// Various prefixes: 0x, -, etc.
	if f.Sharp {
		switch base {
		case 8:
			if buf[i] != '0' {
				i--
				buf[i] = '0'
			}
		case 16:
			// Add a leading 0x or 0X.
			i--
			buf[i] = digits[16]
			i--
			buf[i] = '0'
		}
	}

	if negative {
		i--
		buf[i] = '-'
	} else if f.Plus {
		i--
		buf[i] = '+'
	} else if f.Space {
		i--
		buf[i] = ' '
	}

	// Left padding with zeros has already been handled like precision earlier
	// or the f.zero flag is ignored due to an explicitly set precision.
	oldZero := f.Zero
	f.Zero = false
	f.pad(buf[i:])
	f.Zero = oldZero
}

// truncate truncates the string to the specified precision, if present.
func (f *formatInfo) truncate(s string) string {
	if f.PrecPresent {
		n := f.Prec
		for i := range s {
			n--
			if n < 0 {
				return s[:i]
			}
		}
	}
	return s
}

// fmt_s formats a string.
func (f *formatInfo) fmt_s(s string) {
	s = f.truncate(s)
	f.padString(s)
}

// fmt_sbx formats a string or byte slice as a hexadecimal encoding of its bytes.
func (f *formatInfo) fmt_sbx(s string, b []byte, digits string) {
	length := len(b)
	if b == nil {
		// No byte slice present. Assume string s should be encoded.
		length = len(s)
	}
	// Set length to not process more bytes than the precision demands.
	if f.PrecPresent && f.Prec < length {
		length = f.Prec
	}
	// Compute width of the encoding taking into account the f.sharp and f.space flag.
	width := 2 * length
	if width > 0 {
		if f.Space {
			// Each element encoded by two hexadecimals will get a leading 0x or 0X.
			if f.Sharp {
				width *= 2
			}
			// Elements will be separated by a space.
			width += length - 1
		} else if f.Sharp {
			// Only a leading 0x or 0X will be added for the whole string.
			width += 2
		}
	} else { // The byte slice or string that should be encoded is empty.
		if f.WidthPresent {
			f.writePadding(f.Width)
		}
		return
	}
	// Handle padding to the left.
	if f.WidthPresent && f.Width > width && !f.Minus {
		f.writePadding(f.Width - width)
	}
	// Write the encoding directly into the output buffer.
	buf := f.buf
	if f.Sharp {
		// Add leading 0x or 0X.
		buf.WriteByte('0')
		buf.WriteByte(digits[16])
	}
	var c byte
	for i := 0; i < length; i++ {
		if f.Space && i > 0 {
			// Separate elements with a space.
			buf.WriteByte(' ')
			if f.Sharp {
				// Add leading 0x or 0X for each element.
				buf.WriteByte('0')
				buf.WriteByte(digits[16])
			}
		}
		if b != nil {
			c = b[i] // Take a byte from the input byte slice.
		} else {
			c = s[i] // Take a byte from the input string.
		}
		// Encode each byte as two hexadecimal digits.
		buf.WriteByte(digits[c>>4])
		buf.WriteByte(digits[c&0xF])
	}
	// Handle padding to the right.
	if f.WidthPresent && f.Width > width && f.Minus {
		f.writePadding(f.Width - width)
	}
}

// fmt_sx formats a string as a hexadecimal encoding of its bytes.
func (f *formatInfo) fmt_sx(s, digits string) {
	f.fmt_sbx(s, nil, digits)
}

// fmt_bx formats a byte slice as a hexadecimal encoding of its bytes.
func (f *formatInfo) fmt_bx(b []byte, digits string) {
	f.fmt_sbx("", b, digits)
}

// fmt_q formats a string as a double-quoted, escaped Go string constant.
// If f.sharp is set a raw (backquoted) string may be returned instead
// if the string does not contain any control characters other than tab.
func (f *formatInfo) fmt_q(s string) {
	s = f.truncate(s)
	if f.Sharp && strconv.CanBackquote(s) {
		f.padString("`" + s + "`")
		return
	}
	buf := f.intbuf[:0]
	if f.Plus {
		f.pad(strconv.AppendQuoteToASCII(buf, s))
	} else {
		f.pad(strconv.AppendQuote(buf, s))
	}
}

// fmt_c formats an integer as a Unicode character.
// If the character is not valid Unicode, it will print '\ufffd'.
func (f *formatInfo) fmt_c(c uint64) {
	r := rune(c)
	if c > utf8.MaxRune {
		r = utf8.RuneError
	}
	buf := f.intbuf[:0]
	w := utf8.EncodeRune(buf[:utf8.UTFMax], r)
	f.pad(buf[:w])
}

// fmt_qc formats an integer as a single-quoted, escaped Go character constant.
// If the character is not valid Unicode, it will print '\ufffd'.
func (f *formatInfo) fmt_qc(c uint64) {
	r := rune(c)
	if c > utf8.MaxRune {
		r = utf8.RuneError
	}
	buf := f.intbuf[:0]
	if f.Plus {
		f.pad(strconv.AppendQuoteRuneToASCII(buf, r))
	} else {
		f.pad(strconv.AppendQuoteRune(buf, r))
	}
}

// fmt_float formats a float64. It assumes that verb is a valid format specifier
// for strconv.AppendFloat and therefore fits into a byte.
func (f *formatInfo) fmt_float(v float64, size int, verb rune, prec int) {
	// Explicit precision in format specifier overrules default precision.
	if f.PrecPresent {
		prec = f.Prec
	}
	// Format number, reserving space for leading + sign if needed.
	num := strconv.AppendFloat(f.intbuf[:1], v, byte(verb), prec, size)
	if num[1] == '-' || num[1] == '+' {
		num = num[1:]
	} else {
		num[0] = '+'
	}
	// f.space means to add a leading space instead of a "+" sign unless
	// the sign is explicitly asked for by f.plus.
	if f.Space && num[0] == '+' && !f.Plus {
		num[0] = ' '
	}
	// Special handling for infinities and NaN,
	// which don't look like a number so shouldn't be padded with zeros.
	if num[1] == 'I' || num[1] == 'N' {
		oldZero := f.Zero
		f.Zero = false
		// Remove sign before NaN if not asked for.
		if num[1] == 'N' && !f.Space && !f.Plus {
			num = num[1:]
		}
		f.pad(num)
		f.Zero = oldZero
		return
	}
	// The sharp flag forces printing a decimal point for non-binary formats
	// and retains trailing zeros, which we may need to restore.
	if f.Sharp && verb != 'b' {
		digits := 0
		switch verb {
		case 'v', 'g', 'G':
			digits = prec
			// If no precision is set explicitly use a precision of 6.
			if digits == -1 {
				digits = 6
			}
		}

		// Buffer pre-allocated with enough room for
		// exponent notations of the form "e+123".
		var tailBuf [5]byte
		tail := tailBuf[:0]

		hasDecimalPoint := false
		// Starting from i = 1 to skip sign at num[0].
		for i := 1; i < len(num); i++ {
			switch num[i] {
			case '.':
				hasDecimalPoint = true
			case 'e', 'E':
				tail = append(tail, num[i:]...)
				num = num[:i]
			default:
				digits--
			}
		}
		if !hasDecimalPoint {
			num = append(num, '.')
		}
		for digits > 0 {
			num = append(num, '0')
			digits--
		}
		num = append(num, tail...)
	}
	// We want a sign if asked for and if the sign is not positive.
	if f.Plus || num[0] != '+' {
		// If we're zero padding to the left we want the sign before the leading zeros.
		// Achieve this by writing the sign out and then padding the unsigned number.
		if f.Zero && f.WidthPresent && f.Width > len(num) {
			f.buf.WriteByte(num[0])
			f.writePadding(f.Width - len(num))
			f.buf.Write(num[1:])
			return
		}
		f.pad(num)
		return
	}
	// No sign to show and the number is positive; just print the unsigned number.
	f.pad(num[1:])
}
