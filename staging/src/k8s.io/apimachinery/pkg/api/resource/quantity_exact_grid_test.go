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

package resource

import (
	"math"
	"strconv"
	"strings"
)

// This file holds the parse-law input grid and nothing else. It depends only
// on the standard library so that it can be copied into an older release tree
// to regenerate testdata/quantity_decode_corpus.json (see
// quantity_decode_corpus_test.go).
//
// The grid is a full cross product of sign x mantissa x suffix, so every
// boundary mantissa meets every boundary exponent by construction. The axes
// merge the hand-picked boundaries of the original grid with the axes of
// jpbetz's v1.37.0 decode corpus
// (https://github.com/kubernetes/kubernetes/compare/master...jpbetz:kubernetes:quantity-v137-decode-corpus).

// parseGridSigns is the sign axis.
var parseGridSigns = []string{"", "-", "+"}

// parseGridMantissas is the mantissa axis: digit-less spellings, zeros,
// leading zeros, small integers, the int64 and uint64 rails, the 18- and
// 19-digit precision edge, long mantissas, and fractions around the nano
// boundary and the binary-suffix downgrade.
var parseGridMantissas = []string{
	"", ".", "0", "0.0", "01", "1", "5", "7", "8", "12", "123", "999", "1000", "1024", "123456789",
	"999999999999999999", "1000000000000000000", "10000000000000000000", "100000000000000000000",
	"9223372036854775", "9223372036854776",
	"9223372036854775806", "9223372036854775807", "9223372036854775808",
	"18446744073709551615", "18446744073709551616",
	"123456789012345678901", "1234567890123456789012345678901234567890",
	"1" + strings.Repeat("0", 49), strings.Repeat("9", 100),
	".5", "1.", "0.1", "0.5", "0.9", "1.5", "1.25", "0.001", "0.0001", "0.000000001", "0.0000000001",
	"0.0009765625", "1.0000000001", "123.456789012", "9223372036854775.807", "9223372036854775.808",
}

// parseGridSuffixes is the suffix axis: no suffix, every SI and binary SI
// suffix, the exponent spellings the grammar allows (E, e+, leading zeros),
// and decimal exponents around zero, the nano scale, 64-bit float limits, and
// both int32 edges, plus values that wrap when truncated to 32 bits.
func parseGridSuffixes() []string {
	suffixes := []string{"", "n", "u", "m", "k", "M", "G", "T", "P", "E", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "E3", "e+21", "e03"}
	exponents := []int64{
		-(1 << 32) - 1, -(1 << 32), math.MinInt32 - 1, math.MinInt32, math.MinInt32 + 1, math.MinInt32 + 2,
		math.MinInt32 + 18, -2147483000,
		-1000, -330, -19, -18, -10, -9, -3, -1, 0, 1, 3, 9, 10, 18, 19, 21, 330, 1000,
		2147483000, math.MaxInt32 - 20, math.MaxInt32 - 19, math.MaxInt32 - 1, math.MaxInt32, math.MaxInt32 + 1,
		(1 << 32) - 1, 1 << 32, (1 << 32) + 1, 1 << 33,
	}
	for _, e := range exponents {
		suffixes = append(suffixes, "e"+strconv.FormatInt(e, 10))
	}
	return suffixes
}

// parseGridExtras are inputs outside the product: the digit-count ladder, every
// exponent from -19 to 19 for a few mantissas, the signed int64 rails without a
// suffix, malformed spellings, and exponents far outside 64 bits.
func parseGridExtras() []string {
	var extras []string
	for d := 1; d <= 40; d++ {
		extras = append(extras, "1"+strings.Repeat("0", d-1), strings.Repeat("9", d))
	}
	for _, d := range []int{50, 60, 100} {
		extras = append(extras, "1"+strings.Repeat("0", d-1), strings.Repeat("9", d))
	}
	for e := -19; e <= 19; e++ {
		exp := "e" + strconv.Itoa(e)
		extras = append(extras, "1"+exp, "123"+exp, "-1"+exp)
	}
	extras = append(extras,
		"-9223372036854775808", "-9223372036854775807", "-1000000000000000000", "-10000000000000000000",
		"8Ei", "7Ei", "-8Ei", "1000k", "0Ki", "-0.0001Ki", "0.000000001Ki", "-0.000000001Ki", "-0.0009765625Ki", "-0.9Ki",
		"1K", "1ki", "1KI", "1KiB", "1Zi", "1Z", "1µ", "1μ", "1 Ki", "1i", ".5i", "-3.01i", "0.1mi", "0.1am",
		"1e", "1e+", "1e-", "-3.01e-", "1e1.5", "1e.5", "1e3k", "1E3Ki", "1e+-3",
		"1..5", "1.1.M", "1+1.0M", "1-", "++1", "1,5", "1_000", "0x10", "1/2", "１", "٣", "−1", "1\x00",
		"Inf", "+Inf", "NaN", "Infinity", "aoeu",
		"0e9223372036854775807", "1e9223372036854775807", "0e9223372036854775808", "1e9223372036854775808",
		"0e-9223372036854775808", "1e-9223372036854775808", "0e-9223372036854775809", "1e-9223372036854775809",
		"1e18446744073709551616", "0e99999999999999999999", "1e99999999999999999999",
	)
	return extras
}

// parseGridInputs returns the parse-law inputs in a fixed order, without
// duplicates.
func parseGridInputs() []string {
	seen := map[string]bool{}
	var inputs []string
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			inputs = append(inputs, s)
		}
	}
	suffixes := parseGridSuffixes()
	for _, sign := range parseGridSigns {
		for _, mantissa := range parseGridMantissas {
			for _, suffix := range suffixes {
				add(sign + mantissa + suffix)
			}
		}
	}
	for _, s := range parseGridExtras() {
		add(s)
	}
	return inputs
}
