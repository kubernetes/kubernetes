// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"unicode/utf8"

	"golang.org/x/text/internal/format/plural"
)

// A system identifies a CLDR numbering system.
type system byte

type systemData struct {
	id        system
	digitSize byte              // number of UTF-8 bytes per digit
	zero      [utf8.UTFMax]byte // UTF-8 sequence of zero digit.
}

// A SymbolType identifies a symbol of a specific kind.
type SymbolType int

const (
	SymDecimal SymbolType = iota
	SymGroup
	SymList
	SymPercentSign
	SymPlusSign
	SymMinusSign
	SymExponential
	SymSuperscriptingExponent
	SymPerMille
	SymInfinity
	SymNan
	SymTimeSeparator

	NumSymbolTypes
)

type altSymData struct {
	compactTag uint16
	system     system
	symIndex   byte
}

var countMap = map[string]plural.Form{
	"other": plural.Other,
	"zero":  plural.Zero,
	"one":   plural.One,
	"two":   plural.Two,
	"few":   plural.Few,
	"many":  plural.Many,
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
