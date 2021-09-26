// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package currency

import (
	"fmt"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
)

var (
	cup = MustParseISO("CUP")
	czk = MustParseISO("CZK")
	xcd = MustParseISO("XCD")
	zwr = MustParseISO("ZWR")
)

func TestParseISO(t *testing.T) {
	testCases := []struct {
		in  string
		out Unit
		ok  bool
	}{
		{"USD", USD, true},
		{"xxx", XXX, true},
		{"xts", XTS, true},
		{"XX", XXX, false},
		{"XXXX", XXX, false},
		{"", XXX, false},       // not well-formed
		{"UUU", XXX, false},    // unknown
		{"\u22A9", XXX, false}, // non-ASCII, printable

		{"aaa", XXX, false},
		{"zzz", XXX, false},
		{"000", XXX, false},
		{"999", XXX, false},
		{"---", XXX, false},
		{"\x00\x00\x00", XXX, false},
		{"\xff\xff\xff", XXX, false},
	}
	for i, tc := range testCases {
		if x, err := ParseISO(tc.in); x != tc.out || err == nil != tc.ok {
			t.Errorf("%d:%s: was %s, %v; want %s, %v", i, tc.in, x, err == nil, tc.out, tc.ok)
		}
	}
}

func TestFromRegion(t *testing.T) {
	testCases := []struct {
		region   string
		currency Unit
		ok       bool
	}{
		{"NL", EUR, true},
		{"BE", EUR, true},
		{"AG", xcd, true},
		{"CH", CHF, true},
		{"CU", cup, true},   // first of multiple
		{"DG", USD, true},   // does not have M49 code
		{"150", XXX, false}, // implicit false
		{"CP", XXX, false},  // explicit false in CLDR
		{"CS", XXX, false},  // all expired
		{"ZZ", XXX, false},  // none match
	}
	for _, tc := range testCases {
		cur, ok := FromRegion(language.MustParseRegion(tc.region))
		if cur != tc.currency || ok != tc.ok {
			t.Errorf("%s: got %v, %v; want %v, %v", tc.region, cur, ok, tc.currency, tc.ok)
		}
	}
}

func TestFromTag(t *testing.T) {
	testCases := []struct {
		tag      string
		currency Unit
		conf     language.Confidence
	}{
		{"nl", EUR, language.Low},      // nl also spoken outside Euro land.
		{"nl-BE", EUR, language.Exact}, // region is known
		{"pt", BRL, language.Low},
		{"en", USD, language.Low},
		{"en-u-cu-eur", EUR, language.Exact},
		{"tlh", XXX, language.No}, // Klingon has no country.
		{"es-419", XXX, language.No},
		{"und", USD, language.Low},
	}
	for _, tc := range testCases {
		cur, conf := FromTag(language.MustParse(tc.tag))
		if cur != tc.currency || conf != tc.conf {
			t.Errorf("%s: got %v, %v; want %v, %v", tc.tag, cur, conf, tc.currency, tc.conf)
		}
	}
}

func TestTable(t *testing.T) {
	for i := 4; i < len(currency); i += 4 {
		if a, b := currency[i-4:i-1], currency[i:i+3]; a >= b {
			t.Errorf("currency unordered at element %d: %s >= %s", i, a, b)
		}
	}
	// First currency has index 1, last is numCurrencies.
	if c := currency.Elem(1)[:3]; c != "ADP" {
		t.Errorf("first was %q; want ADP", c)
	}
	if c := currency.Elem(numCurrencies)[:3]; c != "ZWR" {
		t.Errorf("last was %q; want ZWR", c)
	}
}

func TestKindRounding(t *testing.T) {
	testCases := []struct {
		kind  Kind
		cur   Unit
		scale int
		inc   int
	}{
		{Standard, USD, 2, 1},
		{Standard, CHF, 2, 1},
		{Cash, CHF, 2, 5},
		{Standard, TWD, 2, 1},
		{Cash, TWD, 0, 1},
		{Standard, czk, 2, 1},
		{Cash, czk, 0, 1},
		{Standard, zwr, 2, 1},
		{Cash, zwr, 0, 1},
		{Standard, KRW, 0, 1},
		{Cash, KRW, 0, 1}, // Cash defaults to standard.
	}
	for i, tc := range testCases {
		if scale, inc := tc.kind.Rounding(tc.cur); scale != tc.scale && inc != tc.inc {
			t.Errorf("%d: got %d, %d; want %d, %d", i, scale, inc, tc.scale, tc.inc)
		}
	}
}

const body = `package main
import (
	"fmt"
	"golang.org/x/text/currency"
)
func main() {
	%s
}
`

func TestLinking(t *testing.T) {
	t.Skip("skipping flaky test; see golang.org/issue/17538")
	base := getSize(t, `fmt.Print(currency.CLDRVersion)`)
	symbols := getSize(t, `fmt.Print(currency.Symbol(currency.USD))`)
	if d := symbols - base; d < 2*1024 {
		t.Errorf("size(symbols)-size(base) was %d; want > 2K", d)
	}
}

func getSize(t *testing.T, main string) int {
	size, err := testtext.CodeSize(fmt.Sprintf(body, main))
	if err != nil {
		t.Skipf("skipping link size test; binary size could not be determined: %v", err)
	}
	return size
}

func BenchmarkString(b *testing.B) {
	for i := 0; i < b.N; i++ {
		USD.String()
	}
}
