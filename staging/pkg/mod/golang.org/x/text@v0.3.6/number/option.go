// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"fmt"

	"golang.org/x/text/internal/number"
	"golang.org/x/text/language"
)

// An Option configures a Formatter.
type Option option

type option func(tag language.Tag, f *number.Formatter)

// TODO: SpellOut requires support of the ICU RBNF format.
// func SpellOut() Option

// NoSeparator causes a number to be displayed without grouping separators.
func NoSeparator() Option {
	return func(t language.Tag, f *number.Formatter) {
		f.GroupingSize = [2]uint8{}
	}
}

// MaxIntegerDigits limits the number of integer digits, eliminating the
// most significant digits.
func MaxIntegerDigits(max int) Option {
	return func(t language.Tag, f *number.Formatter) {
		if max >= 1<<8 {
			max = (1 << 8) - 1
		}
		f.MaxIntegerDigits = uint8(max)
	}
}

// MinIntegerDigits specifies the minimum number of integer digits, adding
// leading zeros when needed.
func MinIntegerDigits(min int) Option {
	return func(t language.Tag, f *number.Formatter) {
		if min >= 1<<8 {
			min = (1 << 8) - 1
		}
		f.MinIntegerDigits = uint8(min)
	}
}

// MaxFractionDigits specifies the maximum number of fractional digits.
func MaxFractionDigits(max int) Option {
	return func(t language.Tag, f *number.Formatter) {
		if max >= 1<<15 {
			max = (1 << 15) - 1
		}
		f.MaxFractionDigits = int16(max)
	}
}

// MinFractionDigits specifies the minimum number of fractional digits.
func MinFractionDigits(min int) Option {
	return func(t language.Tag, f *number.Formatter) {
		if min >= 1<<8 {
			min = (1 << 8) - 1
		}
		f.MinFractionDigits = uint8(min)
	}
}

// Precision sets the maximum number of significant digits. A negative value
// means exact.
func Precision(prec int) Option {
	return func(t language.Tag, f *number.Formatter) {
		f.SetPrecision(prec)
	}
}

// Scale simultaneously sets MinFractionDigits and MaxFractionDigits to the
// given value.
func Scale(decimals int) Option {
	return func(t language.Tag, f *number.Formatter) {
		f.SetScale(decimals)
	}
}

// IncrementString sets the incremental value to which numbers should be
// rounded. For instance: Increment("0.05") will cause 1.44 to round to 1.45.
// IncrementString also sets scale to the scale of the increment.
func IncrementString(decimal string) Option {
	increment := 0
	scale := 0
	d := decimal
	p := 0
	for ; p < len(d) && '0' <= d[p] && d[p] <= '9'; p++ {
		increment *= 10
		increment += int(d[p]) - '0'
	}
	if p < len(d) && d[p] == '.' {
		for p++; p < len(d) && '0' <= d[p] && d[p] <= '9'; p++ {
			increment *= 10
			increment += int(d[p]) - '0'
			scale++
		}
	}
	if p < len(d) {
		increment = 0
		scale = 0
	}
	return func(t language.Tag, f *number.Formatter) {
		f.Increment = uint32(increment)
		f.IncrementScale = uint8(scale)
		f.SetScale(scale)
	}
}

func noop(language.Tag, *number.Formatter) {}

// PatternOverrides allows users to specify alternative patterns for specific
// languages. The Pattern will be overridden for all languages in a subgroup as
// well. The function will panic for invalid input. It is best to create this
// option at startup time.
// PatternOverrides must be the first Option passed to a formatter.
func PatternOverrides(patterns map[string]string) Option {
	// TODO: make it so that it does not have to be the first option.
	// TODO: use -x-nochild to indicate it does not override child tags.
	m := map[language.Tag]*number.Pattern{}
	for k, v := range patterns {
		tag := language.MustParse(k)
		p, err := number.ParsePattern(v)
		if err != nil {
			panic(fmt.Errorf("number: PatternOverrides: %v", err))
		}
		m[tag] = p
	}
	return func(t language.Tag, f *number.Formatter) {
		// TODO: Use language grouping relation instead of parent relation.
		// TODO: Should parent implement the grouping relation?
		for lang := t; ; lang = t.Parent() {
			if p, ok := m[lang]; ok {
				f.Pattern = *p
				break
			}
			if lang == language.Und {
				break
			}
		}
	}
}

// FormatWidth sets the total format width.
func FormatWidth(n int) Option {
	if n <= 0 {
		return noop
	}
	return func(t language.Tag, f *number.Formatter) {
		f.FormatWidth = uint16(n)
		if f.PadRune == 0 {
			f.PadRune = ' '
		}
	}
}

// Pad sets the rune to be used for filling up to the format width.
func Pad(r rune) Option {
	return func(t language.Tag, f *number.Formatter) {
		f.PadRune = r
	}
}

// TODO:
// - FormatPosition (using type aliasing?)
// - Multiplier: find a better way to represent and figure out what to do
//   with clashes with percent/permille.
// - NumberingSystem(nu string): not accessable in number.Info now. Also, should
//      this be keyed by language or generic?
// - SymbolOverrides(symbols map[string]map[number.SymbolType]string) Option
