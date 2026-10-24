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

// This file holds the value-law cases. The parse-law inputs are in
// quantity_exact_grid_test.go.

import (
	"math"
	"strconv"
	"strings"
)

type valueCase struct {
	id   string
	make func() *Quantity
}

func intstr(n int64) string { return strconv.FormatInt(n, 10) }

// valueCases builds quantities for the value laws, in the int64 backend
// where possible, by parsing for the Dec backend, and via the constructors.
func valueCases() []valueCase {
	var cs []valueCase
	add := func(id string, make func() *Quantity) { cs = append(cs, valueCase{id: id, make: make}) }
	parse := func(id, s string) {
		add(id, func() *Quantity { q := MustParse(s); return &q })
	}

	for d := 1; d <= 40; d++ {
		lo := "1" + strings.Repeat("0", d-1)
		hi := strings.Repeat("9", d)
		parse("digits/"+intstr(int64(d))+"/lo", lo)
		parse("digits/"+intstr(int64(d))+"/hi", hi)
	}
	for _, d := range []int{50, 60, 100} {
		parse("digits/"+intstr(int64(d))+"/lo", "1"+strings.Repeat("0", d-1))
		parse("digits/"+intstr(int64(d))+"/hi", strings.Repeat("9", d))
	}

	for _, e := range []int64{math.MaxInt32 - 19, math.MaxInt32 - 1, math.MaxInt32} {
		parse("exp/1e"+intstr(e), "1e"+intstr(e))
	}
	for e := -19; e <= 19; e++ {
		parse("exp/1e"+intstr(int64(e)), "1e"+intstr(int64(e)))
		parse("exp/123e"+intstr(int64(e)), "123e"+intstr(int64(e)))
	}

	suffixes := []string{"", "n", "u", "m", "k", "M", "G", "T", "P", "E", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"}
	for _, suf := range suffixes {
		parse("suffix/1"+suf, "1"+suf)
		parse("suffix/1.5"+suf, "1.5"+suf)
		parse("suffix/-1.5"+suf, "-1.5"+suf)
	}
	parse("suffix/8Ei", "8Ei")
	parse("suffix/-8Ei", "-8Ei")
	parse("suffix/1024Ki", "1024Ki")
	parse("suffix/0.0001Ki", "0.0001Ki")
	parse("suffix/-0.0001Ki", "-0.0001Ki")
	parse("suffix/-0.0009765625Ki", "-0.0009765625Ki")

	// Constructors.
	for _, v := range []int64{0, 1, -1, 5, -5, 12345, math.MaxInt64, math.MaxInt64 - 1, math.MinInt64, math.MinInt64 + 1} {
		add("ctor/NewQuantity/"+intstr(v), func() *Quantity { return NewQuantity(v, DecimalSI) })
		add("ctor/NewMilliQuantity/"+intstr(v), func() *Quantity { return NewMilliQuantity(v, DecimalSI) })
	}
	for _, sc := range []Scale{0, 1, -1, 2, -2, 3, -3, 6, -6, 9, -9, 12, 15, 18, -18} {
		add("ctor/NewScaledQuantity/7/"+intstr(int64(sc)), func() *Quantity { return NewScaledQuantity(7, sc) })
		add("ctor/NewScaledQuantity/-7/"+intstr(int64(sc)), func() *Quantity { return NewScaledQuantity(-7, sc) })
	}
	// The zero value, never parsed or set, and a quantity with no format:
	// String() treats the empty format as DecimalExponent.
	add("ctor/zero-value", func() *Quantity { return &Quantity{} })
	add("ctor/NewQuantity/1500/no-format", func() *Quantity { return NewQuantity(1500, "") })
	add("ctor/NewMilliQuantity/1500/no-format", func() *Quantity { return NewMilliQuantity(1500, "") })
	// BinarySI built by a constructor rather than parsed: String() downgrades
	// below 1024 and drops to DecimalSI for a fraction.
	for _, v := range []int64{1023, 1024, 1536, -2048, 1 << 62} {
		add("ctor/NewQuantity/"+intstr(v)+"/BinarySI", func() *Quantity { return NewQuantity(v, BinarySI) })
	}
	add("ctor/NewMilliQuantity/1500/BinarySI", func() *Quantity { return NewMilliQuantity(1500, BinarySI) })
	// BinarySI above MaxInt64, reachable only through arithmetic: parsing caps
	// a BinarySI spelling at MaxInt64 (8Ei too), so the spelling String() gives
	// such a value may not parse back to it.
	for _, s := range []string{"8Ei", "-8Ei"} {
		add("mul/capped-"+s+"*1024", func() *Quantity { q := MustParse(s); q.Mul(1024); return &q })
	}
	// Below the float64 range: the nearest float64 is a zero of the value's sign.
	add("ctor/NewScaledQuantity/-7/-400", func() *Quantity { return NewScaledQuantity(-7, -400) })
	add("ctor/NewScaledQuantity/7/MinInt32", func() *Quantity { return NewScaledQuantity(7, math.MinInt32) })
	add("ctor/NewScaledQuantity/MaxInt64/1", func() *Quantity { return NewScaledQuantity(math.MaxInt64, 1) })
	// The positive rail: 100e2147483646 is 1e2147483648, whose canonical exponent
	// does not fit int32, so the spelling has to keep a mantissa of 10.
	add("ctor/NewScaledQuantity/100/2147483646", func() *Quantity { return NewScaledQuantity(100, 2147483646) })

	// ToDec variants of a few small values: the same rational on both backends.
	for _, s := range []string{"5", "1.5", "1000m", "1Ki", "-7", "0.5"} {
		add("todec/"+s, func() *Quantity { q := MustParse(s); q.ToDec(); return &q })
	}

	// Non-canonical spellings the parse fast path caches verbatim; the
	// round-trip law checks that String() emits the canonical form. ("1e0" is
	// already covered by the exp grid above; ".5" is canonicalized correctly.)
	for _, s := range []string{"5.", "1.", "+1", "01", "1E3", "1e+21", ".5"} {
		parse("noncanon/"+s, s)
	}
	return cs
}
