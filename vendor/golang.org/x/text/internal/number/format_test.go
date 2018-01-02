// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"fmt"
	"log"
	"testing"

	"golang.org/x/text/language"
)

func TestAppendDecimal(t *testing.T) {
	type pairs map[string]string // alternates with decimal input and result

	testCases := []struct {
		pattern string
		// We want to be able to test some forms of patterns that cannot be
		// represented as a string.
		pat *Pattern

		test pairs
	}{{
		pattern: "0",
		test: pairs{
			"0":    "0",
			"1":    "1",
			"-1":   "-1",
			".00":  "0",
			"10.":  "10",
			"12":   "12",
			"1.2":  "1",
			"NaN":  "NaN",
			"-Inf": "-∞",
		},
	}, {
		pattern: "+0;+0",
		test: pairs{
			"0":    "+0",
			"1":    "+1",
			"-1":   "-1",
			".00":  "+0",
			"10.":  "+10",
			"12":   "+12",
			"1.2":  "+1",
			"NaN":  "NaN",
			"-Inf": "-∞",
			"Inf":  "+∞",
		},
	}, {
		pattern: "0 +;0 +",
		test: pairs{
			"0":   "0 +",
			"1":   "1 +",
			"-1":  "1 -",
			".00": "0 +",
		},
	}, {
		pattern: "0;0-",
		test: pairs{
			"-1":   "1-",
			"NaN":  "NaN",
			"-Inf": "∞-",
			"Inf":  "∞",
		},
	}, {
		pattern: "0000",
		test: pairs{
			"0":     "0000",
			"1":     "0001",
			"12":    "0012",
			"12345": "12345",
		},
	}, {
		pattern: ".0",
		test: pairs{
			"0":      ".0",
			"1":      "1.0",
			"1.2":    "1.2",
			"1.2345": "1.2",
		},
	}, {
		pattern: "#.0",
		test: pairs{
			"0": ".0",
		},
	}, {
		pattern: "#.0#",
		test: pairs{
			"0": ".0",
			"1": "1.0",
		},
	}, {
		pattern: "0.0#",
		test: pairs{
			"0": "0.0",
		},
	}, {
		pattern: "#0.###",
		test: pairs{
			"0":        "0",
			"1":        "1",
			"1.2":      "1.2",
			"1.2345":   "1.234", // rounding should have been done earlier
			"1234.5":   "1234.5",
			"1234.567": "1234.567",
		},
	}, {
		pattern: "#0.######",
		test: pairs{
			"0":           "0",
			"1234.5678":   "1234.5678",
			"0.123456789": "0.123456",
			"NaN":         "NaN",
			"Inf":         "∞",
		},

		// Test separators.
	}, {
		pattern: "#,#.00",
		test: pairs{
			"100": "1,0,0.00",
		},
	}, {
		pattern: "#,0.##",
		test: pairs{
			"10": "1,0",
		},
	}, {
		pattern: "#,0",
		test: pairs{
			"10": "1,0",
		},
	}, {
		pattern: "#,##,#.00",
		test: pairs{
			"1000": "1,00,0.00",
		},
	}, {
		pattern: "#,##0.###",
		test: pairs{
			"0":           "0",
			"1234.5678":   "1,234.567",
			"0.123456789": "0.123",
		},
	}, {
		pattern: "#,##,##0.###",
		test: pairs{
			"0":            "0",
			"123456789012": "1,23,45,67,89,012",
			"0.123456789":  "0.123",
		},
	}, {
		pattern: "0,00,000.###",
		test: pairs{
			"0":            "0,00,000",
			"123456789012": "1,23,45,67,89,012",
			"12.3456789":   "0,00,012.345",
			"0.123456789":  "0,00,000.123",
		},

		// Support for ill-formed patterns.
	}, {
		pattern: "#",
		test: pairs{
			".00": "", // This is the behavior of fmt.
			"0":   "", // This is the behavior of fmt.
			"1":   "1",
			"10.": "10",
		},
	}, {
		pattern: ".#",
		test: pairs{
			"0":      "", // This is the behavior of fmt.
			"1":      "1",
			"1.2":    "1.2",
			"1.2345": "1.2",
		},
	}, {
		pattern: "#,#.##",
		test: pairs{
			"10": "1,0",
		},
	}, {
		pattern: "#,#",
		test: pairs{
			"10": "1,0",
		},

		// Special patterns
	}, {
		pattern: "#,max_int=2",
		pat: &Pattern{
			MaxIntegerDigits: 2,
		},
		test: pairs{
			"2017": "17",
		},
	}, {
		pattern: "0,max_int=2",
		pat: &Pattern{
			MaxIntegerDigits: 2,
			MinIntegerDigits: 1,
		},
		test: pairs{
			"2000": "0",
			"2001": "1",
			"2017": "17",
		},
	}, {
		pattern: "00,max_int=2",
		pat: &Pattern{
			MaxIntegerDigits: 2,
			MinIntegerDigits: 2,
		},
		test: pairs{
			"2000": "00",
			"2001": "01",
			"2017": "17",
		},
	}, {
		pattern: "@@@@,max_int=2",
		pat: &Pattern{
			MaxIntegerDigits:     2,
			MinSignificantDigits: 4,
		},
		test: pairs{
			"2017": "17.00",
			"2000": "0.000",
			"2001": "1.000",
		},

		// Significant digits
	}, {
		pattern: "@@##",
		test: pairs{
			"1":     "1.0",
			"0.1":   "0.10",
			"123":   "123",
			"1234":  "1234",
			"12345": "12340",
		},
	}, {
		pattern: "@@@@",
		test: pairs{
			"1":     "1.000",
			".1":    "0.1000",
			".001":  "0.001000",
			"123":   "123.0",
			"1234":  "1234",
			"12345": "12340", // rounding down
			"NaN":   "NaN",
			"-Inf":  "-∞",
		},

		// TODO: rounding
		// {"@@@@": "23456": "23460"}, // rounding up
		// TODO: padding

		// Scientific and Engineering notation
	}, {
		pattern: "#E0",
		test: pairs{
			"0":       "0\u202f×\u202f10⁰",
			"1":       "1\u202f×\u202f10⁰",
			"123.456": "1\u202f×\u202f10²",
		},
	}, {
		pattern: "#E+0",
		test: pairs{
			"0":      "0\u202f×\u202f10⁺⁰",
			"1000":   "1\u202f×\u202f10⁺³",
			"1E100":  "1\u202f×\u202f10⁺¹⁰⁰",
			"1E-100": "1\u202f×\u202f10⁻¹⁰⁰",
			"NaN":    "NaN",
			"-Inf":   "-∞",
		},
	}, {
		pattern: "##0E00",
		test: pairs{
			"100":     "100\u202f×\u202f10⁰⁰",
			"12345":   "10\u202f×\u202f10⁰³",
			"123.456": "100\u202f×\u202f10⁰⁰",
		},
	}, {
		pattern: "##0.###E00",
		test: pairs{
			"100":     "100\u202f×\u202f10⁰⁰",
			"12345":   "12.34\u202f×\u202f10⁰³",
			"123.456": "123.4\u202f×\u202f10⁰⁰",
		},
	}, {
		pattern: "##0.000E00",
		test: pairs{
			"100":     "100.0\u202f×\u202f10⁰⁰",
			"12345":   "12.34\u202f×\u202f10⁰³",
			"123.456": "123.4\u202f×\u202f10⁰⁰",
		},
	}, {
		pattern: "@@E0",
		test: pairs{
			"0":    "0.0\u202f×\u202f10⁰",
			"99":   "9.9\u202f×\u202f10¹",
			"0.99": "9.9\u202f×\u202f10⁻¹",
		},
	}, {
		pattern: "@###E00",
		test: pairs{
			"0":     "0\u202f×\u202f10⁰⁰",
			"1":     "1\u202f×\u202f10⁰⁰",
			"11":    "1.1\u202f×\u202f10⁰¹",
			"111":   "1.11\u202f×\u202f10⁰²",
			"1111":  "1.111\u202f×\u202f10⁰³",
			"11111": "1.111\u202f×\u202f10⁰⁴",
			"0.1":   "1\u202f×\u202f10⁻⁰¹",
			"0.11":  "1.1\u202f×\u202f10⁻⁰¹",
			"0.001": "1\u202f×\u202f10⁻⁰³",
		},
	}, {
		pattern: "*x##0",
		test: pairs{
			"0":    "xx0",
			"10":   "x10",
			"100":  "100",
			"1000": "1000",
		},
	}, {
		pattern: "##0*x",
		test: pairs{
			"0":    "0xx",
			"10":   "10x",
			"100":  "100",
			"1000": "1000",
		},
	}, {
		pattern: "* ###0.000",
		test: pairs{
			"0":        "   0.000",
			"123":      " 123.000",
			"123.456":  " 123.456",
			"1234.567": "1234.567",
		},
	}, {
		pattern: "**0.0#######E00",
		test: pairs{
			"0":     "***0.0\u202f×\u202f10⁰⁰",
			"10":    "***1.0\u202f×\u202f10⁰¹",
			"11":    "***1.1\u202f×\u202f10⁰¹",
			"111":   "**1.11\u202f×\u202f10⁰²",
			"1111":  "*1.111\u202f×\u202f10⁰³",
			"11111": "1.1111\u202f×\u202f10⁰⁴",
			"11110": "*1.111\u202f×\u202f10⁰⁴",
			"11100": "**1.11\u202f×\u202f10⁰⁴",
			"11000": "***1.1\u202f×\u202f10⁰⁴",
			"10000": "***1.0\u202f×\u202f10⁰⁴",
		},
	}, {
		pattern: "*xpre0suf",
		test: pairs{
			"0":  "pre0suf",
			"10": "pre10suf",
		},
	}, {
		pattern: "*∞ pre ###0 suf",
		test: pairs{
			"0":    "∞∞∞ pre 0 suf",
			"10":   "∞∞ pre 10 suf",
			"100":  "∞ pre 100 suf",
			"1000": " pre 1000 suf",
		},
	}, {
		pattern: "pre *∞###0 suf",
		test: pairs{
			"0":    "pre ∞∞∞0 suf",
			"10":   "pre ∞∞10 suf",
			"100":  "pre ∞100 suf",
			"1000": "pre 1000 suf",
		},
	}, {
		pattern: "pre ###0*∞ suf",
		test: pairs{
			"0":    "pre 0∞∞∞ suf",
			"10":   "pre 10∞∞ suf",
			"100":  "pre 100∞ suf",
			"1000": "pre 1000 suf",
		},
	}, {
		pattern: "pre ###0 suf *∞",
		test: pairs{
			"0":    "pre 0 suf ∞∞∞",
			"10":   "pre 10 suf ∞∞",
			"100":  "pre 100 suf ∞",
			"1000": "pre 1000 suf ",
		},
	}, {
		// Take width of positive pattern.
		pattern: "**###0;**-#####0x",
		test: pairs{
			"0":  "***0",
			"-1": "*-1x",
		},
	}, {
		pattern: "0.00%",
		test: pairs{
			"0.1": "10.00%",
		},
	}, {
		pattern: "0.##%",
		test: pairs{
			"0.1":     "10%",
			"0.11":    "11%",
			"0.111":   "11.1%",
			"0.1111":  "11.11%",
			"0.11111": "11.11%",
		},
	}, {
		pattern: "‰ 0.0#",
		test: pairs{
			"0.1":      "‰ 100.0",
			"0.11":     "‰ 110.0",
			"0.111":    "‰ 111.0",
			"0.1111":   "‰ 111.1",
			"0.11111":  "‰ 111.11",
			"0.111111": "‰ 111.11",
		},
	}}

	// TODO:
	// 	"#,##0.00¤",
	// 	"#,##0.00 ¤;(#,##0.00 ¤)",

	for _, tc := range testCases {
		pat := tc.pat
		if pat == nil {
			var err error
			if pat, err = ParsePattern(tc.pattern); err != nil {
				log.Fatal(err)
			}
		}
		var f Formatter
		f.InitPattern(language.English, pat)
		for dec, want := range tc.test {
			buf := make([]byte, 100)
			t.Run(tc.pattern+"/"+dec, func(t *testing.T) {
				dec := mkdec(dec)
				buf = f.Format(buf[:0], &dec)
				if got := string(buf); got != want {
					t.Errorf("\n got %[1]q (%[1]s)\nwant %[2]q (%[2]s)", got, want)
				}
			})
		}
	}
}

func TestLocales(t *testing.T) {
	testCases := []struct {
		tag  language.Tag
		num  string
		want string
	}{
		{language.Make("en"), "123456.78", "123,456.78"},
		{language.Make("de"), "123456.78", "123.456,78"},
		{language.Make("de-CH"), "123456.78", "123’456.78"},
		{language.Make("fr"), "123456.78", "123 456,78"},
		{language.Make("bn"), "123456.78", "১,২৩,৪৫৬.৭৮"},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprint(tc.tag, "/", tc.num), func(t *testing.T) {
			var f Formatter
			f.InitDecimal(tc.tag)
			d := mkdec(tc.num)
			b := f.Format(nil, &d)
			if got := string(b); got != tc.want {
				t.Errorf("got %[1]q (%[1]s); want %[2]q (%[2]s)", got, tc.want)
			}
		})
	}
}

func TestFormatters(t *testing.T) {
	var f Formatter
	testCases := []struct {
		init func(t language.Tag)
		num  string
		want string
	}{
		{f.InitDecimal, "123456.78", "123,456.78"},
		{f.InitScientific, "123456.78", "1.23\u202f×\u202f10⁵"},
		{f.InitEngineering, "123456.78", "123\u202f×\u202f10³"},

		{f.InitPercent, "0.1234", "12.34%"},
		{f.InitPerMille, "0.1234", "123.40‰"},
	}
	for i, tc := range testCases {
		t.Run(fmt.Sprint(i, "/", tc.num), func(t *testing.T) {
			tc.init(language.English)
			f.Pattern.MinFractionDigits = 2
			f.Pattern.MaxFractionDigits = 2
			d := mkdec(tc.num)
			b := f.Format(nil, &d)
			if got := string(b); got != tc.want {
				t.Errorf("got %[1]q (%[1]s); want %[2]q (%[2]s)", got, tc.want)
			}
		})
	}
}
