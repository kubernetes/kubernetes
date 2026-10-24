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
	"fmt"
	"math/big"
)

// String returns q in canonical form in its own format.
func (q Quantity) String() string {
	return q.Canonical(q.Format())
}

// Canonical renders q in the canonical form the resource.Quantity doc comment
// describes: no fractional digits and the largest exponent or suffix that
// loses nothing. The value decides the digits; the format only picks the
// suffix.
func (q Quantity) Canonical(format string) string {
	if q.IsZero() {
		return "0"
	}
	d := q.decimal()
	f := format
	if f == BinarySI {
		switch {
		case cmpMag(q, NewInt64(1024, 0)) < 0 || d.exponent < 0:
			// Below 1024 or fractional, BinarySI prints as DecimalSI.
			f = DecimalSI
		case d.exponent < 70:
			// 10^70 already has seven factors of 1024, one past Ei.
			n := new(big.Int).Mul(d.mantissa, pow10(d.exponent))
			if k := n.TrailingZeroBits() / 10; k <= 6 {
				return n.Rsh(n, 10*k).String() + [...]string{"", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"}[k]
			}
			f = DecimalExponent
		default:
			f = DecimalExponent
		}
	}
	num, exp := new(big.Int).Set(d.mantissa), d.exponent
	for exp%3 != 0 {
		num.Mul(num, bigTen)
		exp--
	}
	si := map[int64]string{-9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E"}
	if s, ok := si[exp]; ok && f == DecimalSI {
		return num.String() + s
	}
	if exp == 0 {
		return num.String()
	}
	return fmt.Sprintf("%se%d", num, exp)
}
