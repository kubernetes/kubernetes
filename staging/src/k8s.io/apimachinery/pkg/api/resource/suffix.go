/*
Copyright 2014 The Kubernetes Authors.

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
	"math"
	"strconv"
)

// suffixValue represents a quantity suffix, which has a multiplier value and a Format.
type suffixValue struct {
	format Format
	// decimalScale is the value multiplier for decimal suffixes, represented as
	// a 10^n scale value. It's the exponent value that was parsed, capped to fit
	// the widenedScale range. Note that this can be outside the Scale range. It
	// must not be saturated to the Scale range until after offsetting it for any
	// digits after the decimal point from the parsed quantity.
	decimalScale widenedScale
	// binaryExponent is the value multiplier for BinarySI suffixes (1024^n).
	binaryExponent int32
}

// parseSuffix returns the value of the suffix s, or false if s isn't a valid suffix.
func parseSuffix(s string) (suffixValue, bool) {
	switch s {
	case "n":
		return suffixValue{format: DecimalSI, decimalScale: -9}, true
	case "u":
		return suffixValue{format: DecimalSI, decimalScale: -6}, true
	case "m":
		return suffixValue{format: DecimalSI, decimalScale: -3}, true
	case "":
		return suffixValue{format: DecimalSI}, true
	case "k":
		return suffixValue{format: DecimalSI, decimalScale: 3}, true
	case "M":
		return suffixValue{format: DecimalSI, decimalScale: 6}, true
	case "G":
		return suffixValue{format: DecimalSI, decimalScale: 9}, true
	case "T":
		return suffixValue{format: DecimalSI, decimalScale: 12}, true
	case "P":
		return suffixValue{format: DecimalSI, decimalScale: 15}, true
	case "E":
		return suffixValue{format: DecimalSI, decimalScale: 18}, true
	case "Ki":
		return suffixValue{format: BinarySI, binaryExponent: 1}, true
	case "Mi":
		return suffixValue{format: BinarySI, binaryExponent: 2}, true
	case "Gi":
		return suffixValue{format: BinarySI, binaryExponent: 3}, true
	case "Ti":
		return suffixValue{format: BinarySI, binaryExponent: 4}, true
	case "Pi":
		return suffixValue{format: BinarySI, binaryExponent: 5}, true
	case "Ei":
		return suffixValue{format: BinarySI, binaryExponent: 6}, true
	}

	if len(s) > 1 && (s[0] == 'E' || s[0] == 'e') {
		parsed, err := strconv.ParseInt(s[1:], 10, 64)
		if err != nil {
			return suffixValue{}, false
		}
		// Clamp so scale arithmetic can't overflow.
		scale := widenedScale(min(max(parsed, math.MinInt32), 2*math.MaxInt32))
		return suffixValue{format: DecimalExponent, decimalScale: scale}, true
	}

	return suffixValue{}, false
}

// decimalSISuffixes holds the DecimalSI suffixes.
var decimalSISuffixes = [...][]byte{
	[]byte("n"), []byte("u"), []byte("m"), []byte(""), []byte("k"),
	[]byte("M"), []byte("G"), []byte("T"), []byte("P"), []byte("E"),
}

// decimalSISuffix returns the DecimalSI suffix for 10^exponent, or false otherwise.
func decimalSISuffix(exponent Scale) ([]byte, bool) {
	if exponent < Nano || exponent > Exa || exponent%3 != 0 {
		return nil, false
	}
	return decimalSISuffixes[(exponent-Nano)/3], true
}

// binarySISuffixes holds the BinarySI suffixes (1024^n).
var binarySISuffixes = [...][]byte{
	nil, []byte("Ki"), []byte("Mi"), []byte("Gi"), []byte("Ti"), []byte("Pi"), []byte("Ei"),
}

// binarySISuffix returns the BinarySI suffix for 1024^exponent, or false otherwise.
func binarySISuffix(exponent int32) ([]byte, bool) {
	if exponent < 0 || int(exponent) >= len(binarySISuffixes) {
		return nil, false
	}
	return binarySISuffixes[exponent], true
}

// decimalExponentSuffix returns the DecimalExponent suffix for 10^exponent, such as "e3".
func decimalExponentSuffix(exponent Scale) []byte {
	if exponent == 0 {
		return nil
	}
	// Eight bytes fits most exponents.
	result := append(make([]byte, 0, 8), 'e')
	return strconv.AppendInt(result, int64(exponent), 10)
}
