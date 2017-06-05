// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package currency

import (
	"fmt"
	"io"
	"sort"

	"golang.org/x/text/internal"
	"golang.org/x/text/internal/format"
	"golang.org/x/text/language"
)

// Amount is an amount-currency unit pair.
type Amount struct {
	amount   interface{} // Change to decimal(64|128).
	currency Unit
}

// Currency reports the currency unit of this amount.
func (a Amount) Currency() Unit { return a.currency }

// TODO: based on decimal type, but may make sense to customize a bit.
// func (a Amount) Decimal()
// func (a Amount) Int() (int64, error)
// func (a Amount) Fraction() (int64, error)
// func (a Amount) Rat() *big.Rat
// func (a Amount) Float() (float64, error)
// func (a Amount) Scale() uint
// func (a Amount) Precision() uint
// func (a Amount) Sign() int
//
// Add/Sub/Div/Mul/Round.

var space = []byte(" ")

// Format implements fmt.Formatter. It accepts format.State for
// language-specific rendering.
func (a Amount) Format(s fmt.State, verb rune) {
	v := formattedValue{
		currency: a.currency,
		amount:   a.amount,
		format:   defaultFormat,
	}
	v.Format(s, verb)
}

// formattedValue is currency amount or unit that implements language-sensitive
// formatting.
type formattedValue struct {
	currency Unit
	amount   interface{} // Amount, Unit, or number.
	format   *options
}

// Format implements fmt.Formatter. It accepts format.State for
// language-specific rendering.
func (v formattedValue) Format(s fmt.State, verb rune) {
	var lang int
	if state, ok := s.(format.State); ok {
		lang, _ = language.CompactIndex(state.Language())
	}

	// Get the options. Use DefaultFormat if not present.
	opt := v.format
	if opt == nil {
		opt = defaultFormat
	}
	cur := v.currency
	if cur.index == 0 {
		cur = opt.currency
	}

	// TODO: use pattern.
	io.WriteString(s, opt.symbol(lang, cur))
	if v.amount != nil {
		s.Write(space)

		// TODO: apply currency-specific rounding
		scale, _ := opt.kind.Rounding(cur)
		if _, ok := s.Precision(); !ok {
			fmt.Fprintf(s, "%.*f", scale, v.amount)
		} else {
			fmt.Fprint(s, v.amount)
		}
	}
}

// Formatter decorates a given number, Unit or Amount with formatting options.
type Formatter func(amount interface{}) formattedValue

// func (f Formatter) Options(opts ...Option) Formatter

// TODO: call this a Formatter or FormatFunc?

var dummy = USD.Amount(0)

// adjust creates a new Formatter based on the adjustments of fn on f.
func (f Formatter) adjust(fn func(*options)) Formatter {
	var o options = *(f(dummy).format)
	fn(&o)
	return o.format
}

// Default creates a new Formatter that defaults to currency unit c if a numeric
// value is passed that is not associated with a currency.
func (f Formatter) Default(currency Unit) Formatter {
	return f.adjust(func(o *options) { o.currency = currency })
}

// Kind sets the kind of the underlying currency unit.
func (f Formatter) Kind(k Kind) Formatter {
	return f.adjust(func(o *options) { o.kind = k })
}

var defaultFormat *options = ISO(dummy).format

var (
	// Uses Narrow symbols. Overrides Symbol, if present.
	NarrowSymbol Formatter = Formatter(formNarrow)

	// Use Symbols instead of ISO codes, when available.
	Symbol Formatter = Formatter(formSymbol)

	// Use ISO code as symbol.
	ISO Formatter = Formatter(formISO)

	// TODO:
	// // Use full name as symbol.
	// Name Formatter
)

// options configures rendering and rounding options for an Amount.
type options struct {
	currency Unit
	kind     Kind

	symbol func(compactIndex int, c Unit) string
}

func (o *options) format(amount interface{}) formattedValue {
	v := formattedValue{format: o}
	switch x := amount.(type) {
	case Amount:
		v.amount = x.amount
		v.currency = x.currency
	case *Amount:
		v.amount = x.amount
		v.currency = x.currency
	case Unit:
		v.currency = x
	case *Unit:
		v.currency = *x
	default:
		if o.currency.index == 0 {
			panic("cannot format number without a currency being set")
		}
		// TODO: Must be a number.
		v.amount = x
		v.currency = o.currency
	}
	return v
}

var (
	optISO    = options{symbol: lookupISO}
	optSymbol = options{symbol: lookupSymbol}
	optNarrow = options{symbol: lookupNarrow}
)

// These need to be functions, rather than curried methods, as curried methods
// are evaluated at init time, causing tables to be included unconditionally.
func formISO(x interface{}) formattedValue    { return optISO.format(x) }
func formSymbol(x interface{}) formattedValue { return optSymbol.format(x) }
func formNarrow(x interface{}) formattedValue { return optNarrow.format(x) }

func lookupISO(x int, c Unit) string    { return c.String() }
func lookupSymbol(x int, c Unit) string { return normalSymbol.lookup(x, c) }
func lookupNarrow(x int, c Unit) string { return narrowSymbol.lookup(x, c) }

type symbolIndex struct {
	index []uint16 // position corresponds with compact index of language.
	data  []curToIndex
}

var (
	normalSymbol = symbolIndex{normalLangIndex, normalSymIndex}
	narrowSymbol = symbolIndex{narrowLangIndex, narrowSymIndex}
)

func (x *symbolIndex) lookup(lang int, c Unit) string {
	for {
		index := x.data[x.index[lang]:x.index[lang+1]]
		i := sort.Search(len(index), func(i int) bool {
			return index[i].cur >= c.index
		})
		if i < len(index) && index[i].cur == c.index {
			x := index[i].idx
			start := x + 1
			end := start + uint16(symbols[x])
			if start == end {
				return c.String()
			}
			return symbols[start:end]
		}
		if lang == 0 {
			break
		}
		lang = int(internal.Parent[lang])
	}
	return c.String()
}
