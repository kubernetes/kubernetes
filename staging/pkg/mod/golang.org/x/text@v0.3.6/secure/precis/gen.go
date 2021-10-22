// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unicode table generator.
// Data read from the web.

//go:build ignore
// +build ignore

package main

import (
	"flag"
	"log"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/triegen"
	"golang.org/x/text/internal/ucd"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/unicode/rangetable"
)

var outputFile = flag.String("output", "tables.go", "output file for generated tables; default tables.go")

var assigned, disallowedRunes *unicode.RangeTable

var runeCategory = map[rune]category{}

var overrides = map[category]category{
	viramaModifier: viramaJoinT,
	greek:          greekJoinT,
	hebrew:         hebrewJoinT,
}

func setCategory(r rune, cat category) {
	if c, ok := runeCategory[r]; ok {
		if override, ok := overrides[c]; cat == joiningT && ok {
			cat = override
		} else {
			log.Fatalf("%U: multiple categories for rune (%v and %v)", r, c, cat)
		}
	}
	runeCategory[r] = cat
}

func init() {
	if numCategories > 1<<propShift {
		log.Fatalf("Number of categories is %d; may at most be %d", numCategories, 1<<propShift)
	}
}

func main() {
	gen.Init()

	// Load data
	runes := []rune{}
	// PrecisIgnorableProperties: https://tools.ietf.org/html/rfc7564#section-9.13
	ucd.Parse(gen.OpenUCDFile("DerivedCoreProperties.txt"), func(p *ucd.Parser) {
		if p.String(1) == "Default_Ignorable_Code_Point" {
			runes = append(runes, p.Rune(0))
		}
	})
	ucd.Parse(gen.OpenUCDFile("PropList.txt"), func(p *ucd.Parser) {
		switch p.String(1) {
		case "Noncharacter_Code_Point":
			runes = append(runes, p.Rune(0))
		}
	})
	// OldHangulJamo: https://tools.ietf.org/html/rfc5892#section-2.9
	ucd.Parse(gen.OpenUCDFile("HangulSyllableType.txt"), func(p *ucd.Parser) {
		switch p.String(1) {
		case "L", "V", "T":
			runes = append(runes, p.Rune(0))
		}
	})

	disallowedRunes = rangetable.New(runes...)
	assigned = rangetable.Assigned(unicode.Version)

	// Load category data.
	runeCategory['l'] = latinSmallL
	ucd.Parse(gen.OpenUCDFile("UnicodeData.txt"), func(p *ucd.Parser) {
		const cccVirama = 9
		if p.Int(ucd.CanonicalCombiningClass) == cccVirama {
			setCategory(p.Rune(0), viramaModifier)
		}
	})
	ucd.Parse(gen.OpenUCDFile("Scripts.txt"), func(p *ucd.Parser) {
		switch p.String(1) {
		case "Greek":
			setCategory(p.Rune(0), greek)
		case "Hebrew":
			setCategory(p.Rune(0), hebrew)
		case "Hiragana", "Katakana", "Han":
			setCategory(p.Rune(0), japanese)
		}
	})

	// Set the rule categories associated with exceptions. This overrides any
	// previously set categories. The original categories are manually
	// reintroduced in the categoryTransitions table.
	for r, e := range exceptions {
		if e.cat != 0 {
			runeCategory[r] = e.cat
		}
	}
	cat := map[string]category{
		"L": joiningL,
		"D": joiningD,
		"T": joiningT,

		"R": joiningR,
	}
	ucd.Parse(gen.OpenUCDFile("extracted/DerivedJoiningType.txt"), func(p *ucd.Parser) {
		switch v := p.String(1); v {
		case "L", "D", "T", "R":
			setCategory(p.Rune(0), cat[v])
		}
	})

	writeTables()
	gen.Repackage("gen_trieval.go", "trieval.go", "precis")
}

type exception struct {
	prop property
	cat  category
}

func init() {
	// Programmatically add the Arabic and Indic digits to the exceptions map.
	// See comment in the exceptions map below why these are marked disallowed.
	for i := rune(0); i <= 9; i++ {
		exceptions[0x0660+i] = exception{
			prop: disallowed,
			cat:  arabicIndicDigit,
		}
		exceptions[0x06F0+i] = exception{
			prop: disallowed,
			cat:  extendedArabicIndicDigit,
		}
	}
}

// The Exceptions class as defined in RFC 5892
// https://tools.ietf.org/html/rfc5892#section-2.6
var exceptions = map[rune]exception{
	0x00DF: {prop: pValid},
	0x03C2: {prop: pValid},
	0x06FD: {prop: pValid},
	0x06FE: {prop: pValid},
	0x0F0B: {prop: pValid},
	0x3007: {prop: pValid},

	// ContextO|J rules are marked as disallowed, taking a "guilty until proven
	// innocent" approach. The main reason for this is that the check for
	// whether a context rule should be applied can be moved to the logic for
	// handing disallowed runes, taken it off the common path. The exception to
	// this rule is for katakanaMiddleDot, as the rule logic is handled without
	// using a rule function.

	// ContextJ (Join control)
	0x200C: {prop: disallowed, cat: zeroWidthNonJoiner},
	0x200D: {prop: disallowed, cat: zeroWidthJoiner},

	// ContextO
	0x00B7: {prop: disallowed, cat: middleDot},
	0x0375: {prop: disallowed, cat: greekLowerNumeralSign},
	0x05F3: {prop: disallowed, cat: hebrewPreceding}, // punctuation Geresh
	0x05F4: {prop: disallowed, cat: hebrewPreceding}, // punctuation Gershayim
	0x30FB: {prop: pValid, cat: katakanaMiddleDot},

	// These are officially ContextO, but the implementation does not require
	// special treatment of these, so we simply mark them as valid.
	0x0660: {prop: pValid},
	0x0661: {prop: pValid},
	0x0662: {prop: pValid},
	0x0663: {prop: pValid},
	0x0664: {prop: pValid},
	0x0665: {prop: pValid},
	0x0666: {prop: pValid},
	0x0667: {prop: pValid},
	0x0668: {prop: pValid},
	0x0669: {prop: pValid},
	0x06F0: {prop: pValid},
	0x06F1: {prop: pValid},
	0x06F2: {prop: pValid},
	0x06F3: {prop: pValid},
	0x06F4: {prop: pValid},
	0x06F5: {prop: pValid},
	0x06F6: {prop: pValid},
	0x06F7: {prop: pValid},
	0x06F8: {prop: pValid},
	0x06F9: {prop: pValid},

	0x0640: {prop: disallowed},
	0x07FA: {prop: disallowed},
	0x302E: {prop: disallowed},
	0x302F: {prop: disallowed},
	0x3031: {prop: disallowed},
	0x3032: {prop: disallowed},
	0x3033: {prop: disallowed},
	0x3034: {prop: disallowed},
	0x3035: {prop: disallowed},
	0x303B: {prop: disallowed},
}

// LetterDigits: https://tools.ietf.org/html/rfc5892#section-2.1
// r in {Ll, Lu, Lo, Nd, Lm, Mn, Mc}.
func isLetterDigits(r rune) bool {
	return unicode.In(r,
		unicode.Ll, unicode.Lu, unicode.Lm, unicode.Lo, // Letters
		unicode.Mn, unicode.Mc, // Modifiers
		unicode.Nd, // Digits
	)
}

func isIdDisAndFreePVal(r rune) bool {
	return unicode.In(r,
		// OtherLetterDigits: https://tools.ietf.org/html/rfc7564#section-9.18
		// r in in {Lt, Nl, No, Me}
		unicode.Lt, unicode.Nl, unicode.No, // Other letters / numbers
		unicode.Me, // Modifiers

		// Spaces: https://tools.ietf.org/html/rfc7564#section-9.14
		// r in in {Zs}
		unicode.Zs,

		// Symbols: https://tools.ietf.org/html/rfc7564#section-9.15
		// r in {Sm, Sc, Sk, So}
		unicode.Sm, unicode.Sc, unicode.Sk, unicode.So,

		// Punctuation: https://tools.ietf.org/html/rfc7564#section-9.16
		// r in {Pc, Pd, Ps, Pe, Pi, Pf, Po}
		unicode.Pc, unicode.Pd, unicode.Ps, unicode.Pe,
		unicode.Pi, unicode.Pf, unicode.Po,
	)
}

// HasCompat: https://tools.ietf.org/html/rfc7564#section-9.17
func hasCompat(r rune) bool {
	return !norm.NFKC.IsNormalString(string(r))
}

// From https://tools.ietf.org/html/rfc5892:
//
// If .cp. .in. Exceptions Then Exceptions(cp);
//   Else If .cp. .in. BackwardCompatible Then BackwardCompatible(cp);
//   Else If .cp. .in. Unassigned Then UNASSIGNED;
//   Else If .cp. .in. ASCII7 Then PVALID;
//   Else If .cp. .in. JoinControl Then CONTEXTJ;
//   Else If .cp. .in. OldHangulJamo Then DISALLOWED;
//   Else If .cp. .in. PrecisIgnorableProperties Then DISALLOWED;
//   Else If .cp. .in. Controls Then DISALLOWED;
//   Else If .cp. .in. HasCompat Then ID_DIS or FREE_PVAL;
//   Else If .cp. .in. LetterDigits Then PVALID;
//   Else If .cp. .in. OtherLetterDigits Then ID_DIS or FREE_PVAL;
//   Else If .cp. .in. Spaces Then ID_DIS or FREE_PVAL;
//   Else If .cp. .in. Symbols Then ID_DIS or FREE_PVAL;
//   Else If .cp. .in. Punctuation Then ID_DIS or FREE_PVAL;
//   Else DISALLOWED;

func writeTables() {
	propTrie := triegen.NewTrie("derivedProperties")
	w := gen.NewCodeWriter()
	defer w.WriteVersionedGoFile(*outputFile, "precis")
	gen.WriteUnicodeVersion(w)

	// Iterate over all the runes...
	for i := rune(0); i < unicode.MaxRune; i++ {
		r := rune(i)

		if !utf8.ValidRune(r) {
			continue
		}

		e, ok := exceptions[i]
		p := e.prop
		switch {
		case ok:
		case !unicode.In(r, assigned):
			p = unassigned
		case r >= 0x0021 && r <= 0x007e: // Is ASCII 7
			p = pValid
		case unicode.In(r, disallowedRunes, unicode.Cc):
			p = disallowed
		case hasCompat(r):
			p = idDisOrFreePVal
		case isLetterDigits(r):
			p = pValid
		case isIdDisAndFreePVal(r):
			p = idDisOrFreePVal
		default:
			p = disallowed
		}
		cat := runeCategory[r]
		// Don't set category for runes that are disallowed.
		if p == disallowed {
			cat = exceptions[r].cat
		}
		propTrie.Insert(r, uint64(p)|uint64(cat))
	}
	sz, err := propTrie.Gen(w)
	if err != nil {
		log.Fatal(err)
	}
	w.Size += sz
}
