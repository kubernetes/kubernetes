// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import (
	"unicode"
	"unicode/utf8"
)

// NewNumericWeighter wraps w to replace individual digits to sort based on their
// numeric value.
//
// Weighter w must have a free primary weight after the primary weight for 9.
// If this is not the case, numeric value will sort at the same primary level
// as the first primary sorting after 9.
func NewNumericWeighter(w Weighter) Weighter {
	getElem := func(s string) Elem {
		elems, _ := w.AppendNextString(nil, s)
		return elems[0]
	}
	nine := getElem("9")

	// Numbers should order before zero, but the DUCET has no room for this.
	// TODO: move before zero once we use fractional collation elements.
	ns, _ := MakeElem(nine.Primary()+1, nine.Secondary(), int(nine.Tertiary()), 0)

	return &numericWeighter{
		Weighter: w,

		// We assume that w sorts digits of different kinds in order of numeric
		// value and that the tertiary weight order is preserved.
		//
		// TODO: evaluate whether it is worth basing the ranges on the Elem
		// encoding itself once the move to fractional weights is complete.
		zero:          getElem("0"),
		zeroSpecialLo: getElem("０"), // U+FF10 FULLWIDTH DIGIT ZERO
		zeroSpecialHi: getElem("₀"), // U+2080 SUBSCRIPT ZERO
		nine:          nine,
		nineSpecialHi: getElem("₉"), // U+2089 SUBSCRIPT NINE
		numberStart:   ns,
	}
}

// A numericWeighter translates a stream of digits into a stream of weights
// representing the numeric value.
type numericWeighter struct {
	Weighter

	// The Elems below all demarcate boundaries of specific ranges. With the
	// current element encoding digits are in two ranges: normal (default
	// tertiary value) and special. For most languages, digits have collation
	// elements in the normal range.
	//
	// Note: the range tests are very specific for the element encoding used by
	// this implementation. The tests in collate_test.go are designed to fail
	// if this code is not updated when an encoding has changed.

	zero          Elem // normal digit zero
	zeroSpecialLo Elem // special digit zero, low tertiary value
	zeroSpecialHi Elem // special digit zero, high tertiary value
	nine          Elem // normal digit nine
	nineSpecialHi Elem // special digit nine
	numberStart   Elem
}

// AppendNext calls the namesake of the underlying weigher, but replaces single
// digits with weights representing their value.
func (nw *numericWeighter) AppendNext(buf []Elem, s []byte) (ce []Elem, n int) {
	ce, n = nw.Weighter.AppendNext(buf, s)
	nc := numberConverter{
		elems: buf,
		w:     nw,
		b:     s,
	}
	isZero, ok := nc.checkNextDigit(ce)
	if !ok {
		return ce, n
	}
	// ce might have been grown already, so take it instead of buf.
	nc.init(ce, len(buf), isZero)
	for n < len(s) {
		ce, sz := nw.Weighter.AppendNext(nc.elems, s[n:])
		nc.b = s
		n += sz
		if !nc.update(ce) {
			break
		}
	}
	return nc.result(), n
}

// AppendNextString calls the namesake of the underlying weigher, but replaces
// single digits with weights representing their value.
func (nw *numericWeighter) AppendNextString(buf []Elem, s string) (ce []Elem, n int) {
	ce, n = nw.Weighter.AppendNextString(buf, s)
	nc := numberConverter{
		elems: buf,
		w:     nw,
		s:     s,
	}
	isZero, ok := nc.checkNextDigit(ce)
	if !ok {
		return ce, n
	}
	nc.init(ce, len(buf), isZero)
	for n < len(s) {
		ce, sz := nw.Weighter.AppendNextString(nc.elems, s[n:])
		nc.s = s
		n += sz
		if !nc.update(ce) {
			break
		}
	}
	return nc.result(), n
}

type numberConverter struct {
	w *numericWeighter

	elems    []Elem
	nDigits  int
	lenIndex int

	s string // set if the input was of type string
	b []byte // set if the input was of type []byte
}

// init completes initialization of a numberConverter and prepares it for adding
// more digits. elems is assumed to have a digit starting at oldLen.
func (nc *numberConverter) init(elems []Elem, oldLen int, isZero bool) {
	// Insert a marker indicating the start of a number and a placeholder
	// for the number of digits.
	if isZero {
		elems = append(elems[:oldLen], nc.w.numberStart, 0)
	} else {
		elems = append(elems, 0, 0)
		copy(elems[oldLen+2:], elems[oldLen:])
		elems[oldLen] = nc.w.numberStart
		elems[oldLen+1] = 0

		nc.nDigits = 1
	}
	nc.elems = elems
	nc.lenIndex = oldLen + 1
}

// checkNextDigit reports whether bufNew adds a single digit relative to the old
// buffer. If it does, it also reports whether this digit is zero.
func (nc *numberConverter) checkNextDigit(bufNew []Elem) (isZero, ok bool) {
	if len(nc.elems) >= len(bufNew) {
		return false, false
	}
	e := bufNew[len(nc.elems)]
	if e < nc.w.zeroSpecialLo || nc.w.nine < e {
		// Not a number.
		return false, false
	}
	if e < nc.w.zero {
		if e > nc.w.nineSpecialHi {
			// Not a number.
			return false, false
		}
		if !nc.isDigit() {
			return false, false
		}
		isZero = e <= nc.w.zeroSpecialHi
	} else {
		// This is the common case if we encounter a digit.
		isZero = e == nc.w.zero
	}
	// Test the remaining added collation elements have a zero primary value.
	if n := len(bufNew) - len(nc.elems); n > 1 {
		for i := len(nc.elems) + 1; i < len(bufNew); i++ {
			if bufNew[i].Primary() != 0 {
				return false, false
			}
		}
		// In some rare cases, collation elements will encode runes in
		// unicode.No as a digit. For example Ethiopic digits (U+1369 - U+1371)
		// are not in Nd. Also some digits that clearly belong in unicode.No,
		// like U+0C78 TELUGU FRACTION DIGIT ZERO FOR ODD POWERS OF FOUR, have
		// collation elements indistinguishable from normal digits.
		// Unfortunately, this means we need to make this check for nearly all
		// non-Latin digits.
		//
		// TODO: check the performance impact and find something better if it is
		// an issue.
		if !nc.isDigit() {
			return false, false
		}
	}
	return isZero, true
}

func (nc *numberConverter) isDigit() bool {
	if nc.b != nil {
		r, _ := utf8.DecodeRune(nc.b)
		return unicode.In(r, unicode.Nd)
	}
	r, _ := utf8.DecodeRuneInString(nc.s)
	return unicode.In(r, unicode.Nd)
}

// We currently support a maximum of about 2M digits (the number of primary
// values). Such numbers will compare correctly against small numbers, but their
// comparison against other large numbers is undefined.
//
// TODO: define a proper fallback, such as comparing large numbers textually or
// actually allowing numbers of unlimited length.
//
// TODO: cap this to a lower number (like 100) and maybe allow a larger number
// in an option?
const maxDigits = 1<<maxPrimaryBits - 1

func (nc *numberConverter) update(elems []Elem) bool {
	isZero, ok := nc.checkNextDigit(elems)
	if nc.nDigits == 0 && isZero {
		return true
	}
	nc.elems = elems
	if !ok {
		return false
	}
	nc.nDigits++
	return nc.nDigits < maxDigits
}

// result fills in the length element for the digit sequence and returns the
// completed collation elements.
func (nc *numberConverter) result() []Elem {
	e, _ := MakeElem(nc.nDigits, defaultSecondary, defaultTertiary, 0)
	nc.elems[nc.lenIndex] = e
	return nc.elems
}
