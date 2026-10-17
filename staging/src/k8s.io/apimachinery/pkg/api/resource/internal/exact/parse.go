/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package exact

import (
	"errors"
	"math"
	"math/big"
	"regexp"
	"strings"
)

// Grammar is the regular expression resource.Quantity declares for
// separating a number from its suffix (splitREString, quoted by
// ErrFormatWrong); a test in the resource package keeps the two identical.
// Parse splits its input with it and then checks each half against the
// grammar in the Quantity doc comment, which is stricter: the number needs at
// least one digit and at most one dot, and the suffix must be one of the
// listed ones or a decimal exponent.
const Grammar = "^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$"

var splitRE = regexp.MustCompile(Grammar)

var (
	// ErrFormatWrong is returned for input outside the documented grammar.
	ErrFormatWrong = errors.New("quantity does not match the documented grammar")
	// ErrOutOfDomain is returned for a non-zero quantity whose decimal
	// exponent is above MaxExponent: it is in the grammar, but Parse does not
	// value it.
	ErrOutOfDomain = errors.New("quantity exponent is outside the modelled domain")
)

// suffix is what a suffix multiplies the number by: base^power.
type suffix struct {
	base   int64
	power  int64
	format string
}

// suffixes is the documented suffix table. It is deliberately separate from
// the resource package's suffixer, which is where decimal exponents are
// narrowed to int32 (the #142395 class); sharing it would share that.
var suffixes = map[string]suffix{
	"":   {10, 0, DecimalSI},
	"n":  {10, -9, DecimalSI},
	"u":  {10, -6, DecimalSI},
	"m":  {10, -3, DecimalSI},
	"k":  {10, 3, DecimalSI},
	"M":  {10, 6, DecimalSI},
	"G":  {10, 9, DecimalSI},
	"T":  {10, 12, DecimalSI},
	"P":  {10, 15, DecimalSI},
	"E":  {10, 18, DecimalSI},
	"Ki": {2, 10, BinarySI},
	"Mi": {2, 20, BinarySI},
	"Gi": {2, 30, BinarySI},
	"Ti": {2, 40, BinarySI},
	"Pi": {2, 50, BinarySI},
	"Ei": {2, 60, BinarySI},
}

var decimalExponentRE = regexp.MustCompile(`^[eE]([+-]?)([0-9]+)$`)

// parseSuffix returns what s multiplies the number by. A decimal exponent
// beyond MaxExponent in either direction is returned as one past it, keeping
// its sign.
func parseSuffix(s string) (suffix, bool) {
	if sf, ok := suffixes[s]; ok {
		return sf, true
	}
	m := decimalExponentRE.FindStringSubmatch(s)
	if m == nil {
		return suffix{}, false
	}
	e, _ := new(big.Int).SetString(m[2], 10)
	if m[1] == "-" {
		e.Neg(e)
	}
	switch {
	case e.Cmp(big.NewInt(MaxExponent)) > 0:
		return suffix{10, MaxExponent + 1, DecimalExponent}, true
	case e.Cmp(big.NewInt(-MaxExponent)) < 0:
		return suffix{10, -MaxExponent - 1, DecimalExponent}, true
	}
	return suffix{10, e.Int64(), DecimalExponent}, true
}

// Parse parses s by the grammar in the resource.Quantity doc comment and
// returns its exact value after the two adjustments that comment documents
// for parsing: a non-zero value finer than 1n is rounded away from zero to
// 1n, and a binarySI value is capped at 2^63-1 in magnitude. A binarySI value
// of magnitude below one is kept in DecimalSI format, as the parser does.
func Parse(s string) (Quantity, error) {
	parts := splitRE.FindStringSubmatch(s)
	if parts == nil {
		return Quantity{}, ErrFormatWrong
	}
	number, sfx := parts[1], parts[2]
	neg := false
	switch number[0] {
	case '-':
		neg = true
		number = number[1:]
	case '+':
		number = number[1:]
	}
	whole, fraction, _ := strings.Cut(number, ".")
	if whole+fraction == "" || strings.Contains(fraction, ".") {
		return Quantity{}, ErrFormatWrong
	}
	sf, ok := parseSuffix(sfx)
	if !ok {
		return Quantity{}, ErrFormatWrong
	}
	digits, _ := new(big.Int).SetString(whole+fraction, 10)
	if neg {
		digits.Neg(digits)
	}
	format := sf.format
	var q Quantity
	switch exponent := sf.power - int64(len(fraction)); {
	case digits.Sign() == 0:
		// Zero is exact whatever its exponent, so the domain does not apply.
		return Quantity{format: format}, nil
	case sf.power > MaxExponent:
		return Quantity{}, ErrOutOfDomain
	case sf.base == 2:
		r := new(big.Rat).SetFrac(digits, pow10(int64(len(fraction))))
		q = newRat(r.Mul(r, new(big.Rat).SetInt(new(big.Int).Lsh(big.NewInt(1), uint(sf.power)))), format)
	case exponent >= -RatLimit:
		m, z := stripZeros(digits)
		if exponent > MaxExponent-z {
			return Quantity{}, ErrOutOfDomain
		}
		q = fromDecimal(decimal{mantissa: m, exponent: exponent + z}, format)
	case exponent+decimalDigits(digits) <= -9:
		// Below 10^-9 the value rounds away from zero to exactly 1n; with
		// the exponent beyond -RatLimit there is nothing else to compute.
		return Quantity{rat: big.NewRat(int64(digits.Sign()), 1_000_000_000), format: format}, nil
	default:
		// At least 1n with an exponent below -RatLimit takes more than
		// RatLimit digits, so building the big.Rat costs no more than the
		// input did.
		q = newRat(new(big.Rat).SetFrac(digits, pow10(-exponent)), format)
	}
	q, _ = q.RoundToScale(-9)
	if sf.base == 2 {
		if cmpMag(q, NewInt64(math.MaxInt64, 0)) > 0 {
			q = New(big.NewInt(int64(q.Sign())*math.MaxInt64), 0, format)
		}
		if cmpMag(q, NewInt64(1, 0)) < 0 {
			q.format = DecimalSI
		}
	}
	return q, nil
}

// MustParse is Parse that panics on error.
func MustParse(s string) Quantity {
	q, err := Parse(s)
	if err != nil {
		panic(err)
	}
	return q
}
