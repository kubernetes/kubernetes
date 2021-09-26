// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plural

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

func TestGetIntApprox(t *testing.T) {
	const big = 1234567890
	testCases := []struct {
		digits string
		start  int
		end    int
		nMod   int
		want   int
	}{
		{"123", 0, 1, 1, 1},
		{"123", 0, 2, 1, big},
		{"123", 0, 2, 2, 12},
		{"123", 3, 4, 2, 0},
		{"12345", 3, 4, 2, 4},
		{"40", 0, 1, 2, 4},
		{"1", 0, 7, 2, big},

		{"123", 0, 5, 2, big},
		{"123", 0, 5, 3, big},
		{"123", 0, 5, 4, big},
		{"123", 0, 5, 5, 12300},
		{"123", 0, 5, 6, 12300},
		{"123", 0, 5, 7, 12300},

		// Translation of examples in MatchDigits.
		// Integer parts
		{"123", 0, 3, 3, 123},  // 123
		{"1234", 0, 3, 3, 123}, // 123.4
		{"1", 0, 6, 8, 100000}, // 100000

		// Fraction parts
		{"123", 3, 3, 3, 0},   // 123
		{"1234", 3, 4, 3, 4},  // 123.4
		{"1234", 3, 5, 3, 40}, // 123.40
		{"1", 6, 8, 8, 0},     // 100000.00
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s:%d:%d/%d", tc.digits, tc.start, tc.end, tc.nMod), func(t *testing.T) {
			got := getIntApprox(mkDigits(tc.digits), tc.start, tc.end, tc.nMod, big)
			if got != tc.want {
				t.Errorf("got %d; want %d", got, tc.want)
			}
		})
	}
}

func mkDigits(s string) []byte {
	b := []byte(s)
	for i := range b {
		b[i] -= '0'
	}
	return b
}

func TestValidForms(t *testing.T) {
	testCases := []struct {
		tag  language.Tag
		want []Form
	}{
		{language.AmericanEnglish, []Form{Other, One}},
		{language.Portuguese, []Form{Other, One}},
		{language.Latvian, []Form{Other, Zero, One}},
		{language.Arabic, []Form{Other, Zero, One, Two, Few, Many}},
		{language.Russian, []Form{Other, One, Few, Many}},
	}
	for _, tc := range testCases {
		got := validForms(cardinal, tc.tag)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("validForms(%v): got %v; want %v", tc.tag, got, tc.want)
		}
	}
}

func TestOrdinal(t *testing.T) {
	testPlurals(t, Ordinal, ordinalTests)
}

func TestCardinal(t *testing.T) {
	testPlurals(t, Cardinal, cardinalTests)
}

func testPlurals(t *testing.T, p *Rules, testCases []pluralTest) {
	for _, tc := range testCases {
		for _, loc := range strings.Split(tc.locales, " ") {
			tag := language.MustParse(loc)
			// Test integers
			for _, s := range tc.integer {
				a := strings.Split(s, "~")
				from := parseUint(t, a[0])
				to := from
				if len(a) > 1 {
					to = parseUint(t, a[1])
				}
				for n := from; n <= to; n++ {
					t.Run(fmt.Sprintf("%s/int(%d)", loc, n), func(t *testing.T) {
						if f := p.matchComponents(tag, n, 0, 0); f != Form(tc.form) {
							t.Errorf("matchComponents: got %v; want %v", f, Form(tc.form))
						}
						digits := []byte(fmt.Sprint(n))
						for i := range digits {
							digits[i] -= '0'
						}
						if f := p.MatchDigits(tag, digits, len(digits), 0); f != Form(tc.form) {
							t.Errorf("MatchDigits: got %v; want %v", f, Form(tc.form))
						}
					})
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
					num := fmt.Sprintf("%[1]d.%0[3]*[2]d", n/m, n%m, scale)
					name := fmt.Sprintf("%s:dec(%s)", loc, num)
					t.Run(name, func(t *testing.T) {
						ff := n % m
						tt := ff
						w := scale
						for tt > 0 && tt%10 == 0 {
							w--
							tt /= 10
						}
						if f := p.MatchPlural(tag, n/m, scale, w, ff, tt); f != Form(tc.form) {
							t.Errorf("MatchPlural: got %v; want %v", f, Form(tc.form))
						}
						if f := p.matchComponents(tag, n/m, n%m, scale); f != Form(tc.form) {
							t.Errorf("matchComponents: got %v; want %v", f, Form(tc.form))
						}
						exp := strings.IndexByte(num, '.')
						digits := []byte(strings.Replace(num, ".", "", 1))
						for i := range digits {
							digits[i] -= '0'
						}
						if f := p.MatchDigits(tag, digits, exp, scale); f != Form(tc.form) {
							t.Errorf("MatchDigits: got %v; want %v", f, Form(tc.form))
						}
					})
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
	p := Cardinal
	en := tagToID(language.English)
	zh := tagToID(language.Chinese)
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
	p := Cardinal
	ar := tagToID(language.Arabic)
	lv := tagToID(language.Latvian)
	for i := 0; i < b.N; i++ {
		matchPlural(p, lv, 0, 19, 2)    // 0.19
		matchPlural(p, lv, 11, 0, 3)    // 11.000
		matchPlural(p, lv, 100, 123, 4) // 0.1230
		matchPlural(p, ar, 0, 0, 0)     // 0
		matchPlural(p, ar, 110, 0, 0)   // 110
		matchPlural(p, ar, 99, 99, 2)   // 99.99
	}
}
