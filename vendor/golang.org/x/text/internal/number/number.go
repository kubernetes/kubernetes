// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go gen_common.go

// Package number contains tools and data for formatting numbers.
package number

import (
	"unicode/utf8"

	"golang.org/x/text/internal"
	"golang.org/x/text/language"
)

// Info holds number formatting configuration data.
type Info struct {
	system   systemData // numbering system information
	symIndex byte       // index to symbols
}

// InfoFromLangID returns a Info for the given compact language identifier and
// numbering system identifier. If system is the empty string, the default
// numbering system will be taken for that language.
func InfoFromLangID(compactIndex int, numberSystem string) Info {
	p := langToDefaults[compactIndex]
	// Lookup the entry for the language.
	pSymIndex := byte(0) // Default: Latin, default symbols
	system, ok := systemMap[numberSystem]
	if !ok {
		// Take the value for the default numbering system. This is by far the
		// most common case as an alternative numbering system is hardly used.
		if p&0x80 == 0 {
			pSymIndex = p
		} else {
			// Take the first entry from the alternatives list.
			data := langToAlt[p&^0x80]
			pSymIndex = data.symIndex
			system = data.system
		}
	} else {
		langIndex := compactIndex
		ns := system
	outerLoop:
		for {
			if p&0x80 == 0 {
				if ns == 0 {
					// The index directly points to the symbol data.
					pSymIndex = p
					break
				}
				// Move to the parent and retry.
				langIndex = int(internal.Parent[langIndex])
			}
			// The index points to a list of symbol data indexes.
			for _, e := range langToAlt[p&^0x80:] {
				if int(e.compactTag) != langIndex {
					if langIndex == 0 {
						// The CLDR root defines full symbol information for all
						// numbering systems (even though mostly by means of
						// aliases). This means that we will never fall back to
						// the default of the language. Also, the loop is
						// guaranteed to terminate as a consequence.
						ns = numLatn
						// Fall back to Latin and start from the original
						// language. See
						// http://unicode.org/reports/tr35/#Locale_Inheritance.
						langIndex = compactIndex
					} else {
						// Fall back to parent.
						langIndex = int(internal.Parent[langIndex])
					}
					break
				}
				if e.system == ns {
					pSymIndex = e.symIndex
					break outerLoop
				}
			}
		}
	}
	if int(system) >= len(numSysData) { // algorithmic
		// Will generate ASCII digits in case the user inadvertently calls
		// WriteDigit or Digit on it.
		d := numSysData[0]
		d.id = system
		return Info{
			system:   d,
			symIndex: pSymIndex,
		}
	}
	return Info{
		system:   numSysData[system],
		symIndex: pSymIndex,
	}
}

// InfoFromTag returns a Info for the given language tag.
func InfoFromTag(t language.Tag) Info {
	for {
		if index, ok := language.CompactIndex(t); ok {
			return InfoFromLangID(index, t.TypeForKey("nu"))
		}
		t = t.Parent()
	}
}

// IsDecimal reports if the numbering system can convert decimal to native
// symbols one-to-one.
func (n Info) IsDecimal() bool {
	return int(n.system.id) < len(numSysData)
}

// WriteDigit writes the UTF-8 sequence for n corresponding to the given ASCII
// digit to dst and reports the number of bytes written. dst must be large
// enough to hold the rune (can be up to utf8.UTFMax bytes).
func (n Info) WriteDigit(dst []byte, asciiDigit rune) int {
	copy(dst, n.system.zero[:n.system.digitSize])
	dst[n.system.digitSize-1] += byte(asciiDigit - '0')
	return int(n.system.digitSize)
}

// AppendDigit appends the UTF-8 sequence for n corresponding to the given digit
// to dst and reports the number of bytes written. dst must be large enough to
// hold the rune (can be up to utf8.UTFMax bytes).
func (n Info) AppendDigit(dst []byte, digit byte) []byte {
	dst = append(dst, n.system.zero[:n.system.digitSize]...)
	dst[len(dst)-1] += digit
	return dst
}

// Digit returns the digit for the numbering system for the corresponding ASCII
// value. For example, ni.Digit('3') could return '三'. Note that the argument
// is the rune constant '3', which equals 51, not the integer constant 3.
func (n Info) Digit(asciiDigit rune) rune {
	var x [utf8.UTFMax]byte
	n.WriteDigit(x[:], asciiDigit)
	r, _ := utf8.DecodeRune(x[:])
	return r
}

// Symbol returns the string for the given symbol type.
func (n Info) Symbol(t SymbolType) string {
	return symData.Elem(int(symIndex[n.symIndex][t]))
}

func formatForLang(t language.Tag, index []byte) *Pattern {
	for ; ; t = t.Parent() {
		if x, ok := language.CompactIndex(t); ok {
			return &formats[index[x]]
		}
	}
}
