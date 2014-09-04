/*
Copyright 2014 Google Inc. All rights reserved.

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

package resource

import (
	"errors"
	"math/big"
	"regexp"
	"strings"

	"speter.net/go/exp/math/dec/inf"
)

// Format lists the three possible formattings of a quantity.
type Format string

const (
	DecimalExponent = Format("DecExponent")
	BinarySI        = Format("BinSI")
	DecimalSI       = Format("DecSI")
)

// Quantity is a fixed-point representation of a number.
// It provides convenient marshaling/unmarshaling in JSON and YAML,
// in addition to String() and Int64() accessors.
//
// The serialization format is:
//
// <serialized>      ::= <sign><numeric> | <numeric>
// <numeric>         ::= <digits><exponent> | <digits>.<digits><exponent>
// <sign>            ::= "+" | "-"
// <digits>          ::= <digit> | <digit><digits>
// <digit>           ::= 0 | 1 | ... | 9
// <exponent>        ::= <binarySuffix> | <decimalExponent> | <decimalSuffix>
// <binarySuffix>    ::= i | Ki | Mi | Gi | Ti | Pi | Ei
// <decimalSuffix>   ::= m | "" | k | M | G | T | P | E
// <decimalExponent> ::= "e" <digits> | "E" <digits>
//   (Where digits is always a multiple of 3)
//   (Note that 1024 = 1Ki but 1000 = 1k; I didn't choose the capitalization.)
//
// No matter which of the three exponent forms is used, no quantity may represent
// a number less than .001m or greater than 2^63-1 in magnitude. Numbers that exceed
// a bound will be capped at that bound. (E.g.: 0.0001m will be treated as 0.001m.)
// This may be extended in the future if we require larger or smaller quantities.
//
// Numbers with binary suffixes may not have any fractional part.
//
// Quantities will be serialized in the same format that they were parsed from.
// Before serializing, Quantity will be put in "canonical form".
// This means that Exponent will be adjusted up or down (with a
// corresponding increase or decrease in Mantissa) until one of the
// following is true:
//   a. Binary SI mode: Mantissa mod 1024 is nonzero.
//        Examples: 1Gi 300Mi 6Ki 1001Gi
//        Non-canonical: 1024Gi (Should be 1Ti)
//   b. Decimal SI mode: exponent is greater than 3 and one of Mantissa's three least
//      significant digits is nonzero.
//        Examples: 1G 300M 3K 1001G
//        Non-canonical: 1000G (Should be 1T)
//   c. Decimal SI mode: exponent is less than or equal to zero, and Mantissa has no more
//      than three nonzero decimals. If any decimals are nonzero, three
//      decimals will be emitted.
//        Examples: 5 123.450 1.001 0.045
//        Non-canonical: 1m (should be 0.001)
//   d. Decimal exponent mode: as for decimal SI mode, but using the corresponding
//      "e6" or "E6" form.
//
// The sign will be omitted unless the number is negative.
//
// Note that the quantity will NEVER be represented by a floating point number. That is
// the whole point of this exercise.
//
// Non-canonical values will still parse as long as they are well formed,
// but will be re-emitted in their canonical form. (So always use canonical
// form, or don't diff.)
//
// This format is intended to make it difficult to use these numbers without
// writing some sort of special handling code in the hopes that that will
// cause implementors to also use a fixed point implementation.
type Quantity struct {
	Amount *inf.Dec
	Format
}

const (
	splitREString = "^([+-]?[0123456789.]+)([eEimkKMGTP]*[-+]?[0123456789]*)$"
)

var (
	// splitRE is used to get the various parts of a number.
	splitRE = regexp.MustCompile(splitREString)

	ErrFormatWrong      = errors.New("quantities must match the regular expression '" + splitREString + "'")
	ErrNumeric          = errors.New("unable to parse numeric part of quantity")
	ErrSuffix           = errors.New("unable to parse quantity's suffix")
	ErrFractionalBinary = errors.New("numbers with binary-style SI suffixes can't have fractional parts")
)

// ParseQuantity turns str into a Quantity, or returns an error.
func ParseQuantity(str string) (*Quantity, error) {
	parts := splitRE.FindStringSubmatch(strings.TrimSpace(str))
	if len(parts) != 3 {
		return nil, ErrFormatWrong
	}

	amount := new(inf.Dec)
	if _, ok := amount.SetString(parts[1]); !ok {
		return nil, ErrNumeric
	}

	base, exponent, format, ok := quantitySuffixer.interpret(suffix(parts[2]))
	if !ok {
		return nil, ErrSuffix
	}

	// So that no one but us has to think about suffixes, remove it.
	if base == 10 {
		amount.SetScale(amount.Scale() + inf.Scale(-exponent))
	} else if base == 2 {
		// Detect fractional parts by rounding. There's probably
		// a better way to do this.
		if rounded := new(inf.Dec).Round(amount, 0, inf.RoundFloor); rounded.Cmp(amount) != 0 {
			return nil, ErrFractionalBinary
		}
		if exponent < 0 {
			return nil, ErrFractionalBinary
		}
		// exponent will always be a multiple of 10.
		dec1024 := inf.NewDec(1024, 0)
		for exponent > 0 {
			amount.Mul(amount, dec1024)
			exponent -= 10
		}
	}

	return &Quantity{amount, format}, nil
}

var (
	// Commonly needed big.Ints-- treat as read only!
	ten      = big.NewInt(10)
	zero     = big.NewInt(0)
	thousand = big.NewInt(1000)
	ten24    = big.NewInt(1024)

	minAllowed = inf.NewDec(1, 6)
	maxAllowed = inf.NewDec(999, -18)
)

// removeFactors divides in a loop; the return values have the property that
// d == result * factor ^ times
// d may be modified in place.
// If d == 0, then the return values will be (0, 0)
func removeFactors(d, factor *big.Int) (result *big.Int, times int) {
	q := big.NewInt(0)
	m := big.NewInt(0)
	for d.Cmp(zero) != 0 {
		q.DivMod(d, factor, m)
		if m.Cmp(zero) != 0 {
			break
		}
		times++
		d, q = q, d
	}
	return d, times
}

// Canonicalize returns the canonical form of q and its suffix (see comment on Quantity).
func (q *Quantity) Canonicalize() (string, suffix) {
	mantissa := q.Amount.UnscaledBig()
	exponent := int(-q.Amount.Scale())
	amount := big.NewInt(0).Set(mantissa)

	switch q.Format {
	case DecimalExponent, DecimalSI:
		// move all factors of 10 into the exponent for easy reasoning
		amount, times := removeFactors(amount, ten)
		exponent += times

		// make sure exponent is a multiple of 3
		for exponent%3 != 0 {
			amount.Mul(amount, ten)
			exponent--
		}

		absAmount := big.NewInt(0).Abs(amount)

		// Canonical form has three decimal digits.
		if absAmount.Cmp(thousand) >= 0 {
			// Unless that would cause an exponent of 3-- 111.111e3 is silly.
			if exponent != 0 {
				suffix, _ := quantitySuffixer.construct(10, exponent+3, q.Format)
				number := inf.NewDecBig(amount, 3).String()
				return number, suffix
			}
		}
		suffix, _ := quantitySuffixer.construct(10, exponent, q.Format)
		number := amount.String()
		return number, suffix
	case BinarySI:
		// Apply the (base-10) shift. This will lose any fractional
		// part, which is intentional.
		for exponent < 0 {
			amount.Mul(amount, ten)
			exponent++
		}
		for exponent > 0 {
			amount.Mul(amount, ten)
			exponent--
		}

		amount, exponent := removeFactors(amount, ten24)
		suffix, _ := quantitySuffixer.construct(2, exponent*10, q.Format)
		number := amount.String()
		return number, suffix
	}
	return "0", ""
}

// String formats the Quantity as a string.
func (q *Quantity) String() string {
	number, suffix := q.Canonicalize()
	return number + string(suffix)
}

// MarshalJSON implements the json.Marshaller interface.
func (q Quantity) MarshalJSON() ([]byte, error) {
	return []byte(`"` + q.String() + `"`), nil
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (q *Quantity) UnmarshalJSON(value []byte) error {
	str := string(value)
	parsed, err := ParseQuantity(strings.Trim(str, `"`))
	if err != nil {
		return err
	}
	// This copy is safe because parsed will not be referred to again.
	*q = *parsed
	return nil
}
