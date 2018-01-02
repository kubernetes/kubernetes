// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// Form defines a plural form.
//
// Not all languages support all forms. Also, the meaning of each form varies
// per language. It is important to note that the name of a form does not
// necessarily correspond one-to-one with the set of numbers. For instance,
// for Croation, One matches not only 1, but also 11, 21, etc.
//
// Each language must at least support the form "other".
type Form byte

const (
	Other Form = iota
	Zero
	One
	Two
	Few
	Many
)

var countMap = map[string]Form{
	"other": Other,
	"zero":  Zero,
	"one":   One,
	"two":   Two,
	"few":   Few,
	"many":  Many,
}

type pluralCheck struct {
	// category:
	// 3..7: opID
	// 0..2: category
	cat   byte
	setID byte
}

// opID identifies the type of operand in the plural rule, being i, n or f.
// (v, w, and t are treated as filters in our implementation.)
type opID byte

const (
	opMod           opID = 0x1    // is '%' used?
	opNotEqual      opID = 0x2    // using "!=" to compare
	opI             opID = 0 << 2 // integers after taking the absolute value
	opN             opID = 1 << 2 // full number (must be integer)
	opF             opID = 2 << 2 // fraction
	opV             opID = 3 << 2 // number of visible digits
	opW             opID = 4 << 2 // number of visible digits without trailing zeros
	opBretonM       opID = 5 << 2 // hard-wired rule for Breton
	opItalian800    opID = 6 << 2 // hard-wired rule for Italian
	opAzerbaijan00s opID = 7 << 2 // hard-wired rule for Azerbaijan
)
const (
	// Use this plural form to indicate the next rule needs to match as well.
	// The last condition in the list will have the correct plural form.
	andNext  = 0x7
	formMask = 0x7

	opShift = 3

	// numN indicates the maximum integer, or maximum mod value, for which we
	// have inclusion masks.
	numN = 100
	// The common denominator of the modulo that is taken.
	maxMod = 100
)
