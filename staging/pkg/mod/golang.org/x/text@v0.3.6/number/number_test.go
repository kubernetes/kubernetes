// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"strings"
	"testing"

	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func TestFormatter(t *testing.T) {
	overrides := map[string]string{
		"en": "*e#######0",
		"nl": "*n#######0",
	}
	testCases := []struct {
		desc string
		tag  string
		f    Formatter
		want string
	}{{
		desc: "decimal",
		f:    Decimal(3),
		want: "3",
	}, {
		desc: "decimal fraction",
		f:    Decimal(0.123),
		want: "0.123",
	}, {
		desc: "separators",
		f:    Decimal(1234.567),
		want: "1,234.567",
	}, {
		desc: "no separators",
		f:    Decimal(1234.567, NoSeparator()),
		want: "1234.567",
	}, {
		desc: "max integer",
		f:    Decimal(1973, MaxIntegerDigits(2)),
		want: "73",
	}, {
		desc: "max integer overflow",
		f:    Decimal(1973, MaxIntegerDigits(1000)),
		want: "1,973",
	}, {
		desc: "min integer",
		f:    Decimal(12, MinIntegerDigits(5)),
		want: "00,012",
	}, {
		desc: "max fraction zero",
		f:    Decimal(0.12345, MaxFractionDigits(0)),
		want: "0",
	}, {
		desc: "max fraction 2",
		f:    Decimal(0.12, MaxFractionDigits(2)),
		want: "0.12",
	}, {
		desc: "min fraction 2",
		f:    Decimal(0.12, MaxFractionDigits(2)),
		want: "0.12",
	}, {
		desc: "max fraction overflow",
		f:    Decimal(0.125, MaxFractionDigits(1e6)),
		want: "0.125",
	}, {
		desc: "min integer overflow",
		f:    Decimal(0, MinIntegerDigits(1e6)),
		want: strings.Repeat("000,", 255/3-1) + "000",
	}, {
		desc: "min fraction overflow",
		f:    Decimal(0, MinFractionDigits(1e6)),
		want: "0." + strings.Repeat("0", 255), // TODO: fraction separators
	}, {
		desc: "format width",
		f:    Decimal(123, FormatWidth(10)),
		want: "       123",
	}, {
		desc: "format width pad option before",
		f:    Decimal(123, Pad('*'), FormatWidth(10)),
		want: "*******123",
	}, {
		desc: "format width pad option after",
		f:    Decimal(123, FormatWidth(10), Pad('*')),
		want: "*******123",
	}, {
		desc: "format width illegal",
		f:    Decimal(123, FormatWidth(-1)),
		want: "123",
	}, {
		desc: "increment",
		f:    Decimal(10.33, IncrementString("0.5")),
		want: "10.5",
	}, {
		desc: "increment",
		f:    Decimal(10, IncrementString("ppp")),
		want: "10",
	}, {
		desc: "increment and scale",
		f:    Decimal(10.33, IncrementString("0.5"), Scale(2)),
		want: "10.50",
	}, {
		desc: "pattern overrides en",
		tag:  "en",
		f:    Decimal(101, PatternOverrides(overrides)),
		want: "eeeee101",
	}, {
		desc: "pattern overrides nl",
		tag:  "nl",
		f:    Decimal(101, PatternOverrides(overrides)),
		want: "nnnnn101",
	}, {
		desc: "pattern overrides de",
		tag:  "de",
		f:    Decimal(101, PatternOverrides(overrides)),
		want: "101",
	}, {
		desc: "language selection",
		tag:  "bn",
		f:    Decimal(123456.78, Scale(2)),
		want: "১,২৩,৪৫৬.৭৮",
	}, {
		desc: "scale",
		f:    Decimal(1234.567, Scale(2)),
		want: "1,234.57",
	}, {
		desc: "scientific",
		f:    Scientific(3.00),
		want: "3\u202f×\u202f10⁰",
	}, {
		desc: "scientific",
		f:    Scientific(1234),
		want: "1.234\u202f×\u202f10³",
	}, {
		desc: "scientific",
		f:    Scientific(1234, Scale(2)),
		want: "1.23\u202f×\u202f10³",
	}, {
		desc: "engineering",
		f:    Engineering(12345),
		want: "12.345\u202f×\u202f10³",
	}, {
		desc: "engineering scale",
		f:    Engineering(12345, Scale(2)),
		want: "12.34\u202f×\u202f10³",
	}, {
		desc: "engineering precision(4)",
		f:    Engineering(12345, Precision(4)),
		want: "12.34\u202f×\u202f10³",
	}, {
		desc: "engineering precision(2)",
		f:    Engineering(1234.5, Precision(2)),
		want: "1.2\u202f×\u202f10³",
	}, {
		desc: "percent",
		f:    Percent(0.12),
		want: "12%",
	}, {
		desc: "permille",
		f:    PerMille(0.123),
		want: "123‰",
	}, {
		desc: "percent rounding",
		f:    PerMille(0.12345),
		want: "123‰",
	}, {
		desc: "percent fraction",
		f:    PerMille(0.12345, Scale(2)),
		want: "123.45‰",
	}, {
		desc: "percent fraction",
		f:    PerMille(0.12344, Scale(1)),
		want: "123.4‰",
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			tag := language.Und
			if tc.tag != "" {
				tag = language.MustParse(tc.tag)
			}
			got := message.NewPrinter(tag).Sprint(tc.f)
			if got != tc.want {
				t.Errorf("got %q; want %q", got, tc.want)
			}
		})
	}
}
