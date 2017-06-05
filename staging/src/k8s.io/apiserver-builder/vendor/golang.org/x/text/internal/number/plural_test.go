// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"strconv"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

func TestOrdinal(t *testing.T) {
	testPlurals(t, &ordinalData, ordinalTests)
}

func TestCardinal(t *testing.T) {
	testPlurals(t, &cardinalData, cardinalTests)
}

func testPlurals(t *testing.T, p *pluralRules, testCases []pluralTest) {
	for _, tc := range testCases {
		for _, loc := range strings.Split(tc.locales, " ") {
			langIndex, _ := language.CompactIndex(language.MustParse(loc))
			// Test integers
			for _, s := range tc.integer {
				a := strings.Split(s, "~")
				from := parseUint(t, a[0])
				to := from
				if len(a) > 1 {
					to = parseUint(t, a[1])
				}
				for n := from; n <= to; n++ {
					if f := matchPlural(p, langIndex, n, 0, 0); f != tc.form {
						t.Errorf("%s:int(%d) = %v; want %v", loc, n, f, tc.form)
					}
				}
			}
			// Test decimals
			for _, s := range tc.decimal {
				a := strings.Split(s, "~")
				from, scale := parseFixedPoint(t, a[0])
				to := from
				if len(a) > 1 {
					var toScale int
					if to, toScale = parseFixedPoint(t, a[1]); toScale != scale {
						t.Fatalf("%s:%s: non-matching scales %d versus %d", loc, s, scale, toScale)
					}
				}
				m := 1
				for i := 0; i < scale; i++ {
					m *= 10
				}
				for n := from; n <= to; n++ {
					if f := matchPlural(p, langIndex, n/m, n%m, scale); f != tc.form {
						t.Errorf("%[1]s:dec(%[2]d.%0[4]*[3]d) = %[5]v; want %[6]v", loc, n/m, n%m, scale, f, tc.form)
					}
				}
			}
		}
	}
}

func parseUint(t *testing.T, s string) int {
	val, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		t.Fatal(err)
	}
	return int(val)
}

func parseFixedPoint(t *testing.T, s string) (val, scale int) {
	p := strings.Index(s, ".")
	s = strings.Replace(s, ".", "", 1)
	v, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		t.Fatal(err)
	}
	return int(v), len(s) - p
}

func BenchmarkPluralSimpleCases(b *testing.B) {
	p := &cardinalData
	en, _ := language.CompactIndex(language.English)
	zh, _ := language.CompactIndex(language.Chinese)
	for i := 0; i < b.N; i++ {
		matchPlural(p, en, 0, 0, 0)  // 0
		matchPlural(p, en, 1, 0, 0)  // 1
		matchPlural(p, en, 2, 12, 3) // 2.120
		matchPlural(p, zh, 0, 0, 0)  // 0
		matchPlural(p, zh, 1, 0, 0)  // 1
		matchPlural(p, zh, 2, 12, 3) // 2.120
	}
}

func BenchmarkPluralComplexCases(b *testing.B) {
	p := &cardinalData
	ar, _ := language.CompactIndex(language.Arabic)
	lv, _ := language.CompactIndex(language.Latvian)
	for i := 0; i < b.N; i++ {
		matchPlural(p, lv, 0, 19, 2)    // 0.19
		matchPlural(p, lv, 11, 0, 3)    // 11.000
		matchPlural(p, lv, 100, 123, 4) // 0.1230
		matchPlural(p, ar, 0, 0, 0)     // 0
		matchPlural(p, ar, 110, 0, 0)   // 110
		matchPlural(p, ar, 99, 99, 2)   // 99.99
	}
}
