// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Based on gopkg.in/mgo.v2/bson by Gustavo Niemeyer
// See THIRD-PARTY-NOTICES for original license terms.

package primitive

import (
	"fmt"
	"strconv"
	"strings"
)

// Decimal128 holds decimal128 BSON values.
type Decimal128 struct {
	h, l uint64
}

// NewDecimal128 creates a Decimal128 using the provide high and low uint64s.
func NewDecimal128(h, l uint64) Decimal128 {
	return Decimal128{h: h, l: l}
}

// GetBytes returns the underlying bytes of the BSON decimal value as two uint16 values. The first
// contains the most first 8 bytes of the value and the second contains the latter.
func (d Decimal128) GetBytes() (uint64, uint64) {
	return d.h, d.l
}

// String returns a string representation of the decimal value.
func (d Decimal128) String() string {
	var pos int     // positive sign
	var e int       // exponent
	var h, l uint64 // significand high/low

	if d.h>>63&1 == 0 {
		pos = 1
	}

	switch d.h >> 58 & (1<<5 - 1) {
	case 0x1F:
		return "NaN"
	case 0x1E:
		return "-Infinity"[pos:]
	}

	l = d.l
	if d.h>>61&3 == 3 {
		// Bits: 1*sign 2*ignored 14*exponent 111*significand.
		// Implicit 0b100 prefix in significand.
		e = int(d.h>>47&(1<<14-1)) - 6176
		//h = 4<<47 | d.h&(1<<47-1)
		// Spec says all of these values are out of range.
		h, l = 0, 0
	} else {
		// Bits: 1*sign 14*exponent 113*significand
		e = int(d.h>>49&(1<<14-1)) - 6176
		h = d.h & (1<<49 - 1)
	}

	// Would be handled by the logic below, but that's trivial and common.
	if h == 0 && l == 0 && e == 0 {
		return "-0"[pos:]
	}

	var repr [48]byte // Loop 5 times over 9 digits plus dot, negative sign, and leading zero.
	var last = len(repr)
	var i = len(repr)
	var dot = len(repr) + e
	var rem uint32
Loop:
	for d9 := 0; d9 < 5; d9++ {
		h, l, rem = divmod(h, l, 1e9)
		for d1 := 0; d1 < 9; d1++ {
			// Handle "-0.0", "0.00123400", "-1.00E-6", "1.050E+3", etc.
			if i < len(repr) && (dot == i || l == 0 && h == 0 && rem > 0 && rem < 10 && (dot < i-6 || e > 0)) {
				e += len(repr) - i
				i--
				repr[i] = '.'
				last = i - 1
				dot = len(repr) // Unmark.
			}
			c := '0' + byte(rem%10)
			rem /= 10
			i--
			repr[i] = c
			// Handle "0E+3", "1E+3", etc.
			if l == 0 && h == 0 && rem == 0 && i == len(repr)-1 && (dot < i-5 || e > 0) {
				last = i
				break Loop
			}
			if c != '0' {
				last = i
			}
			// Break early. Works without it, but why.
			if dot > i && l == 0 && h == 0 && rem == 0 {
				break Loop
			}
		}
	}
	repr[last-1] = '-'
	last--

	if e > 0 {
		return string(repr[last+pos:]) + "E+" + strconv.Itoa(e)
	}
	if e < 0 {
		return string(repr[last+pos:]) + "E" + strconv.Itoa(e)
	}
	return string(repr[last+pos:])
}

func divmod(h, l uint64, div uint32) (qh, ql uint64, rem uint32) {
	div64 := uint64(div)
	a := h >> 32
	aq := a / div64
	ar := a % div64
	b := ar<<32 + h&(1<<32-1)
	bq := b / div64
	br := b % div64
	c := br<<32 + l>>32
	cq := c / div64
	cr := c % div64
	d := cr<<32 + l&(1<<32-1)
	dq := d / div64
	dr := d % div64
	return (aq<<32 | bq), (cq<<32 | dq), uint32(dr)
}

var dNaN = Decimal128{0x1F << 58, 0}
var dPosInf = Decimal128{0x1E << 58, 0}
var dNegInf = Decimal128{0x3E << 58, 0}

func dErr(s string) (Decimal128, error) {
	return dNaN, fmt.Errorf("cannot parse %q as a decimal128", s)
}

//ParseDecimal128 takes the given string and attempts to parse it into a valid
// Decimal128 value.
func ParseDecimal128(s string) (Decimal128, error) {
	orig := s
	if s == "" {
		return dErr(orig)
	}
	neg := s[0] == '-'
	if neg || s[0] == '+' {
		s = s[1:]
	}

	if (len(s) == 3 || len(s) == 8) && (s[0] == 'N' || s[0] == 'n' || s[0] == 'I' || s[0] == 'i') {
		if s == "NaN" || s == "nan" || strings.EqualFold(s, "nan") {
			return dNaN, nil
		}
		if s == "Inf" || s == "inf" || strings.EqualFold(s, "inf") || strings.EqualFold(s, "infinity") {
			if neg {
				return dNegInf, nil
			}
			return dPosInf, nil
		}
		return dErr(orig)
	}

	var h, l uint64
	var e int

	var add, ovr uint32
	var mul uint32 = 1
	var dot = -1
	var digits = 0
	var i = 0
	for i < len(s) {
		c := s[i]
		if mul == 1e9 {
			h, l, ovr = muladd(h, l, mul, add)
			mul, add = 1, 0
			if ovr > 0 || h&((1<<15-1)<<49) > 0 {
				return dErr(orig)
			}
		}
		if c >= '0' && c <= '9' {
			i++
			if c > '0' || digits > 0 {
				digits++
			}
			if digits > 34 {
				if c == '0' {
					// Exact rounding.
					e++
					continue
				}
				return dErr(orig)
			}
			mul *= 10
			add *= 10
			add += uint32(c - '0')
			continue
		}
		if c == '.' {
			i++
			if dot >= 0 || i == 1 && len(s) == 1 {
				return dErr(orig)
			}
			if i == len(s) {
				break
			}
			if s[i] < '0' || s[i] > '9' || e > 0 {
				return dErr(orig)
			}
			dot = i
			continue
		}
		break
	}
	if i == 0 {
		return dErr(orig)
	}
	if mul > 1 {
		h, l, ovr = muladd(h, l, mul, add)
		if ovr > 0 || h&((1<<15-1)<<49) > 0 {
			return dErr(orig)
		}
	}
	if dot >= 0 {
		e += dot - i
	}
	if i+1 < len(s) && (s[i] == 'E' || s[i] == 'e') {
		i++
		eneg := s[i] == '-'
		if eneg || s[i] == '+' {
			i++
			if i == len(s) {
				return dErr(orig)
			}
		}
		n := 0
		for i < len(s) && n < 1e4 {
			c := s[i]
			i++
			if c < '0' || c > '9' {
				return dErr(orig)
			}
			n *= 10
			n += int(c - '0')
		}
		if eneg {
			n = -n
		}
		e += n
		for e < -6176 {
			// Subnormal.
			var div uint32 = 1
			for div < 1e9 && e < -6176 {
				div *= 10
				e++
			}
			var rem uint32
			h, l, rem = divmod(h, l, div)
			if rem > 0 {
				return dErr(orig)
			}
		}
		for e > 6111 {
			// Clamped.
			var mul uint32 = 1
			for mul < 1e9 && e > 6111 {
				mul *= 10
				e--
			}
			h, l, ovr = muladd(h, l, mul, 0)
			if ovr > 0 || h&((1<<15-1)<<49) > 0 {
				return dErr(orig)
			}
		}
		if e < -6176 || e > 6111 {
			return dErr(orig)
		}
	}

	if i < len(s) {
		return dErr(orig)
	}

	h |= uint64(e+6176) & uint64(1<<14-1) << 49
	if neg {
		h |= 1 << 63
	}
	return Decimal128{h, l}, nil
}

func muladd(h, l uint64, mul uint32, add uint32) (resh, resl uint64, overflow uint32) {
	mul64 := uint64(mul)
	a := mul64 * (l & (1<<32 - 1))
	b := a>>32 + mul64*(l>>32)
	c := b>>32 + mul64*(h&(1<<32-1))
	d := c>>32 + mul64*(h>>32)

	a = a&(1<<32-1) + uint64(add)
	b = b&(1<<32-1) + a>>32
	c = c&(1<<32-1) + b>>32
	d = d&(1<<32-1) + c>>32

	return (d<<32 | c&(1<<32-1)), (b<<32 | a&(1<<32-1)), uint32(d >> 32)
}
