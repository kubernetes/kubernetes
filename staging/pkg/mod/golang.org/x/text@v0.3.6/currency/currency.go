// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go gen_common.go -output tables.go

// Package currency contains currency-related functionality.
//
// NOTE: the formatting functionality is currently under development and may
// change without notice.
package currency // import "golang.org/x/text/currency"

import (
	"errors"
	"sort"

	"golang.org/x/text/internal/tag"
	"golang.org/x/text/language"
)

// TODO:
// - language-specific currency names.
// - currency formatting.
// - currency information per region
// - register currency code (there are no private use area)

// TODO: remove Currency type from package language.

// Kind determines the rounding and rendering properties of a currency value.
type Kind struct {
	rounding rounding
	// TODO: formatting type: standard, accounting. See CLDR.
}

type rounding byte

const (
	standard rounding = iota
	cash
)

var (
	// Standard defines standard rounding and formatting for currencies.
	Standard Kind = Kind{rounding: standard}

	// Cash defines rounding and formatting standards for cash transactions.
	Cash Kind = Kind{rounding: cash}

	// Accounting defines rounding and formatting standards for accounting.
	Accounting Kind = Kind{rounding: standard}
)

// Rounding reports the rounding characteristics for the given currency, where
// scale is the number of fractional decimals and increment is the number of
// units in terms of 10^(-scale) to which to round to.
func (k Kind) Rounding(cur Unit) (scale, increment int) {
	info := currency.Elem(int(cur.index))[3]
	switch k.rounding {
	case standard:
		info &= roundMask
	case cash:
		info >>= cashShift
	}
	return int(roundings[info].scale), int(roundings[info].increment)
}

// Unit is an ISO 4217 currency designator.
type Unit struct {
	index uint16
}

// String returns the ISO code of u.
func (u Unit) String() string {
	if u.index == 0 {
		return "XXX"
	}
	return currency.Elem(int(u.index))[:3]
}

// Amount creates an Amount for the given currency unit and amount.
func (u Unit) Amount(amount interface{}) Amount {
	// TODO: verify amount is a supported number type
	return Amount{amount: amount, currency: u}
}

var (
	errSyntax = errors.New("currency: tag is not well-formed")
	errValue  = errors.New("currency: tag is not a recognized currency")
)

// ParseISO parses a 3-letter ISO 4217 currency code. It returns an error if s
// is not well-formed or not a recognized currency code.
func ParseISO(s string) (Unit, error) {
	var buf [4]byte // Take one byte more to detect oversize keys.
	key := buf[:copy(buf[:], s)]
	if !tag.FixCase("XXX", key) {
		return Unit{}, errSyntax
	}
	if i := currency.Index(key); i >= 0 {
		if i == xxx {
			return Unit{}, nil
		}
		return Unit{uint16(i)}, nil
	}
	return Unit{}, errValue
}

// MustParseISO is like ParseISO, but panics if the given currency unit
// cannot be parsed. It simplifies safe initialization of Unit values.
func MustParseISO(s string) Unit {
	c, err := ParseISO(s)
	if err != nil {
		panic(err)
	}
	return c
}

// FromRegion reports the currency unit that is currently legal tender in the
// given region according to CLDR. It will return false if region currently does
// not have a legal tender.
func FromRegion(r language.Region) (currency Unit, ok bool) {
	x := regionToCode(r)
	i := sort.Search(len(regionToCurrency), func(i int) bool {
		return regionToCurrency[i].region >= x
	})
	if i < len(regionToCurrency) && regionToCurrency[i].region == x {
		return Unit{regionToCurrency[i].code}, true
	}
	return Unit{}, false
}

// FromTag reports the most likely currency for the given tag. It considers the
// currency defined in the -u extension and infers the region if necessary.
func FromTag(t language.Tag) (Unit, language.Confidence) {
	if cur := t.TypeForKey("cu"); len(cur) == 3 {
		c, _ := ParseISO(cur)
		return c, language.Exact
	}
	r, conf := t.Region()
	if cur, ok := FromRegion(r); ok {
		return cur, conf
	}
	return Unit{}, language.No
}

var (
	// Undefined and testing.
	XXX Unit = Unit{}
	XTS Unit = Unit{xts}

	// G10 currencies https://en.wikipedia.org/wiki/G10_currencies.
	USD Unit = Unit{usd}
	EUR Unit = Unit{eur}
	JPY Unit = Unit{jpy}
	GBP Unit = Unit{gbp}
	CHF Unit = Unit{chf}
	AUD Unit = Unit{aud}
	NZD Unit = Unit{nzd}
	CAD Unit = Unit{cad}
	SEK Unit = Unit{sek}
	NOK Unit = Unit{nok}

	// Additional common currencies as defined by CLDR.
	BRL Unit = Unit{brl}
	CNY Unit = Unit{cny}
	DKK Unit = Unit{dkk}
	INR Unit = Unit{inr}
	RUB Unit = Unit{rub}
	HKD Unit = Unit{hkd}
	IDR Unit = Unit{idr}
	KRW Unit = Unit{krw}
	MXN Unit = Unit{mxn}
	PLN Unit = Unit{pln}
	SAR Unit = Unit{sar}
	THB Unit = Unit{thb}
	TRY Unit = Unit{try}
	TWD Unit = Unit{twd}
	ZAR Unit = Unit{zar}

	// Precious metals.
	XAG Unit = Unit{xag}
	XAU Unit = Unit{xau}
	XPT Unit = Unit{xpt}
	XPD Unit = Unit{xpd}
)
