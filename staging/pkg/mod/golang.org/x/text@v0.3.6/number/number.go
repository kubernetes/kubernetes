// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

// TODO:
//    p.Printf("The gauge was at %v.", number.Spell(number.Percent(23)))
//    // Prints: The gauge was at twenty-three percent.
//
//    p.Printf("From here to %v!", number.Spell(math.Inf()))
//    // Prints: From here to infinity!
//

import (
	"golang.org/x/text/internal/number"
)

const (
	decimalVerbs    = "vfgd"
	scientificVerbs = "veg"
)

// Decimal formats a number as a floating point decimal.
func Decimal(x interface{}, opts ...Option) Formatter {
	return newFormatter(decimalOptions, opts, x)
}

var decimalOptions = newOptions(decimalVerbs, (*number.Formatter).InitDecimal)

// Scientific formats a number in scientific format.
func Scientific(x interface{}, opts ...Option) Formatter {
	return newFormatter(scientificOptions, opts, x)
}

var scientificOptions = newOptions(scientificVerbs, (*number.Formatter).InitScientific)

// Engineering formats a number using engineering notation, which is like
// scientific notation, but with the exponent normalized to multiples of 3.
func Engineering(x interface{}, opts ...Option) Formatter {
	return newFormatter(engineeringOptions, opts, x)
}

var engineeringOptions = newOptions(scientificVerbs, (*number.Formatter).InitEngineering)

// Percent formats a number as a percentage. A value of 1.0 means 100%.
func Percent(x interface{}, opts ...Option) Formatter {
	return newFormatter(percentOptions, opts, x)
}

var percentOptions = newOptions(decimalVerbs, (*number.Formatter).InitPercent)

// PerMille formats a number as a per mille indication. A value of 1.0 means
// 1000‰.
func PerMille(x interface{}, opts ...Option) Formatter {
	return newFormatter(perMilleOptions, opts, x)
}

var perMilleOptions = newOptions(decimalVerbs, (*number.Formatter).InitPerMille)

// TODO:
// - Shortest: akin to verb 'g' of 'G'
//
// TODO: RBNF forms:
// - Compact: 1M 3.5T
// - CompactBinary: 1Mi 3.5Ti
// - Long: 1 million
// - Ordinal:
// - Roman: MCMIIXX
// - RomanSmall: mcmiixx
// - Text: numbers as it typically appears in running text, allowing
//   language-specific choices for when to use numbers and when to use words.
// - Spell?: spelled-out number. Maybe just allow as an option?

// NOTE: both spelled-out numbers and ordinals, to render correctly, need
// detailed linguistic information from the translated string into which they
// are substituted. We will need to implement that first.
