// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"bytes"
	"flag"
	"fmt"
	"strings"
	"testing"
)

var verbose = flag.Bool("verbose", false, "set to true to print the internal tables of matchers")

func TestAddLikelySubtags(t *testing.T) {
	tests := []struct{ in, out string }{
		{"aa", "aa-Latn-ET"},
		{"aa-Latn", "aa-Latn-ET"},
		{"aa-Arab", "aa-Arab-ET"},
		{"aa-Arab-ER", "aa-Arab-ER"},
		{"kk", "kk-Cyrl-KZ"},
		{"kk-CN", "kk-Arab-CN"},
		{"cmn", "cmn"},
		{"zh-AU", "zh-Hant-AU"},
		{"zh-VN", "zh-Hant-VN"},
		{"zh-SG", "zh-Hans-SG"},
		{"zh-Hant", "zh-Hant-TW"},
		{"zh-Hani", "zh-Hani-CN"},
		{"und-Hani", "zh-Hani-CN"},
		{"und", "en-Latn-US"},
		{"und-GB", "en-Latn-GB"},
		{"und-CW", "pap-Latn-CW"},
		{"und-YT", "fr-Latn-YT"},
		{"und-Arab", "ar-Arab-EG"},
		{"und-AM", "hy-Armn-AM"},
		{"und-002", "en-Latn-NG"},
		{"und-Latn-002", "en-Latn-NG"},
		{"en-Latn-002", "en-Latn-NG"},
		{"en-002", "en-Latn-NG"},
		{"en-001", "en-Latn-US"},
		{"und-003", "en-Latn-US"},
		{"und-GB", "en-Latn-GB"},
		{"Latn-001", "en-Latn-US"},
		{"en-001", "en-Latn-US"},
		{"es-419", "es-Latn-419"},
		{"he-145", "he-Hebr-IL"},
		{"ky-145", "ky-Latn-TR"},
		{"kk", "kk-Cyrl-KZ"},
		// Don't specialize duplicate and ambiguous matches.
		{"kk-034", "kk-Arab-034"}, // Matches IR and AF. Both are Arab.
		{"ku-145", "ku-Latn-TR"},  // Matches IQ, TR, and LB, but kk -> TR.
		{"und-Arab-CC", "ms-Arab-CC"},
		{"und-Arab-GB", "ks-Arab-GB"},
		{"und-Hans-CC", "zh-Hans-CC"},
		{"und-CC", "en-Latn-CC"},
		{"sr", "sr-Cyrl-RS"},
		{"sr-151", "sr-Latn-151"}, // Matches RO and RU.
		// We would like addLikelySubtags to generate the same results if the input
		// only changes by adding tags that would otherwise have been added
		// by the expansion.
		// In other words:
		//     und-AA -> xx-Scrp-AA   implies und-Scrp-AA -> xx-Scrp-AA
		//     und-AA -> xx-Scrp-AA   implies xx-AA -> xx-Scrp-AA
		//     und-Scrp -> xx-Scrp-AA implies und-Scrp-AA -> xx-Scrp-AA
		//     und-Scrp -> xx-Scrp-AA implies xx-Scrp -> xx-Scrp-AA
		//     xx -> xx-Scrp-AA       implies xx-Scrp -> xx-Scrp-AA
		//     xx -> xx-Scrp-AA       implies xx-AA -> xx-Scrp-AA
		//
		// The algorithm specified in
		//   http://unicode.org/reports/tr35/tr35-9.html#Supplemental_Data,
		// Section C.10, does not handle the first case. For example,
		// the CLDR data contains an entry und-BJ -> fr-Latn-BJ, but not
		// there is no rule for und-Latn-BJ.  According to spec, und-Latn-BJ
		// would expand to en-Latn-BJ, violating the aforementioned principle.
		// We deviate from the spec by letting und-Scrp-AA expand to xx-Scrp-AA
		// if a rule of the form und-AA -> xx-Scrp-AA is defined.
		// Note that as of version 23, CLDR has some explicitly specified
		// entries that do not conform to these rules. The implementation
		// will not correct these explicit inconsistencies. A later versions of CLDR
		// is supposed to fix this.
		{"und-Latn-BJ", "fr-Latn-BJ"},
		{"und-Bugi-ID", "bug-Bugi-ID"},
		// regions, scripts and languages without definitions
		{"und-Arab-AA", "ar-Arab-AA"},
		{"und-Afak-RE", "fr-Afak-RE"},
		{"und-Arab-GB", "ks-Arab-GB"},
		{"abp-Arab-GB", "abp-Arab-GB"},
		// script has preference over region
		{"und-Arab-NL", "ar-Arab-NL"},
		{"zza", "zza-Latn-TR"},
		// preserve variants and extensions
		{"de-1901", "de-Latn-DE-1901"},
		{"de-x-abc", "de-Latn-DE-x-abc"},
		{"de-1901-x-abc", "de-Latn-DE-1901-x-abc"},
		{"x-abc", "x-abc"}, // TODO: is this the desired behavior?
	}
	for i, tt := range tests {
		in, _ := Parse(tt.in)
		out, _ := Parse(tt.out)
		in, _ = in.addLikelySubtags()
		if in.String() != out.String() {
			t.Errorf("%d: add(%s) was %s; want %s", i, tt.in, in, tt.out)
		}
	}
}
func TestMinimize(t *testing.T) {
	tests := []struct{ in, out string }{
		{"aa", "aa"},
		{"aa-Latn", "aa"},
		{"aa-Latn-ET", "aa"},
		{"aa-ET", "aa"},
		{"aa-Arab", "aa-Arab"},
		{"aa-Arab-ER", "aa-Arab-ER"},
		{"aa-Arab-ET", "aa-Arab"},
		{"und", "und"},
		{"und-Latn", "und"},
		{"und-Latn-US", "und"},
		{"en-Latn-US", "en"},
		{"cmn", "cmn"},
		{"cmn-Hans", "cmn-Hans"},
		{"cmn-Hant", "cmn-Hant"},
		{"zh-AU", "zh-AU"},
		{"zh-VN", "zh-VN"},
		{"zh-SG", "zh-SG"},
		{"zh-Hant", "zh-Hant"},
		{"zh-Hant-TW", "zh-TW"},
		{"zh-Hans", "zh"},
		{"zh-Hani", "zh-Hani"},
		{"und-Hans", "und-Hans"},
		{"und-Hani", "und-Hani"},

		{"und-CW", "und-CW"},
		{"und-YT", "und-YT"},
		{"und-Arab", "und-Arab"},
		{"und-AM", "und-AM"},
		{"und-Arab-CC", "und-Arab-CC"},
		{"und-CC", "und-CC"},
		{"und-Latn-BJ", "und-BJ"},
		{"und-Bugi-ID", "und-Bugi"},
		{"bug-Bugi-ID", "bug-Bugi"},
		// regions, scripts and languages without definitions
		{"und-Arab-AA", "und-Arab-AA"},
		// preserve variants and extensions
		{"de-Latn-1901", "de-1901"},
		{"de-Latn-x-abc", "de-x-abc"},
		{"de-DE-1901-x-abc", "de-1901-x-abc"},
		{"x-abc", "x-abc"}, // TODO: is this the desired behavior?
	}
	for i, tt := range tests {
		in, _ := Parse(tt.in)
		out, _ := Parse(tt.out)
		min, _ := in.minimize()
		if min.String() != out.String() {
			t.Errorf("%d: min(%s) was %s; want %s", i, tt.in, min, tt.out)
		}
		max, _ := min.addLikelySubtags()
		if x, _ := in.addLikelySubtags(); x.String() != max.String() {
			t.Errorf("%d: max(min(%s)) = %s; want %s", i, tt.in, max, x)
		}
	}
}

func TestRegionDistance(t *testing.T) {
	tests := []struct {
		a, b string
		d    int
	}{
		{"NL", "NL", 0},
		{"NL", "EU", 1},
		{"EU", "NL", 1},
		{"005", "005", 0},
		{"NL", "BE", 2},
		{"CO", "005", 1},
		{"005", "CO", 1},
		{"CO", "419", 2},
		{"419", "CO", 2},
		{"005", "419", 1},
		{"419", "005", 1},
		{"001", "013", 2},
		{"013", "001", 2},
		{"CO", "CW", 4},
		{"CO", "PW", 6},
		{"CO", "BV", 6},
		{"ZZ", "QQ", 2},
	}
	for i, tt := range tests {
		ra, _ := getRegionID([]byte(tt.a))
		rb, _ := getRegionID([]byte(tt.b))
		if d := regionDistance(ra, rb); d != tt.d {
			t.Errorf("%d: d(%s, %s) = %v; want %v", i, tt.a, tt.b, d, tt.d)
		}
	}
}

func TestParentDistance(t *testing.T) {
	tests := []struct {
		parent string
		tag    string
		d      uint8
	}{
		{"en-001", "en-AU", 1},
		{"pt-PT", "pt-AO", 1},
		{"pt", "pt-AO", 2},
		{"en-AU", "en-GB", 255},
		{"en-NL", "en-AU", 255},
		// Note that pt-BR and en-US are not automatically minimized.
		{"pt-BR", "pt-AO", 255},
		{"en-US", "en-AU", 255},
	}
	for _, tt := range tests {
		r := Raw.MustParse(tt.parent).region
		tag := Raw.MustParse(tt.tag)
		if d := parentDistance(r, tag); d != tt.d {
			t.Errorf("d(%s, %s) was %d; want %d", r, tag, d, tt.d)
		}
	}
}

// Implementation of String methods for various types for debugging purposes.

func (m *matcher) String() string {
	w := &bytes.Buffer{}
	fmt.Fprintln(w, "Default:", m.default_)
	for tag, h := range m.index {
		fmt.Fprintf(w, "  %s: %v\n", tag, h)
	}
	return w.String()
}

func (h *matchHeader) String() string {
	w := &bytes.Buffer{}
	fmt.Fprintf(w, "exact: ")
	for _, h := range h.exact {
		fmt.Fprintf(w, "%v, ", h)
	}
	fmt.Fprint(w, "; max: ")
	for _, h := range h.max {
		fmt.Fprintf(w, "%v, ", h)
	}
	return w.String()
}

func (t haveTag) String() string {
	return fmt.Sprintf("%v:%d:%v:%v-%v|%v", t.tag, t.index, t.conf, t.maxRegion, t.maxScript, t.altScript)
}

// The test set for TestBestMatch is defined in data_test.go.
func TestBestMatch(t *testing.T) {
	parse := func(list string) (out []Tag) {
		for _, s := range strings.Split(list, ",") {
			out = append(out, mk(strings.TrimSpace(s)))
		}
		return out
	}
	for i, tt := range matchTests {
		supported := parse(tt.supported)
		m := newMatcher(supported)
		if *verbose {
			fmt.Printf("%s:\n%v\n", tt.comment, m)
		}
		for _, tm := range tt.test {
			tag, _, conf := m.Match(parse(tm.desired)...)
			if tag.String() != tm.match {
				t.Errorf("%d:%s: find %s in %q: have %s; want %s (%v)\n", i, tt.comment, tm.desired, tt.supported, tag, tm.match, conf)
			}
		}
	}
}

var benchHave = []Tag{
	mk("en"),
	mk("en-GB"),
	mk("za"),
	mk("zh-Hant"),
	mk("zh-Hans-CN"),
	mk("zh"),
	mk("zh-HK"),
	mk("ar-MK"),
	mk("en-CA"),
	mk("fr-CA"),
	mk("fr-US"),
	mk("fr-CH"),
	mk("fr"),
	mk("lt"),
	mk("lv"),
	mk("iw"),
	mk("iw-NL"),
	mk("he"),
	mk("he-IT"),
	mk("tlh"),
	mk("ja"),
	mk("ja-Jpan"),
	mk("ja-Jpan-JP"),
	mk("de"),
	mk("de-CH"),
	mk("de-AT"),
	mk("de-DE"),
	mk("sr"),
	mk("sr-Latn"),
	mk("sr-Cyrl"),
	mk("sr-ME"),
}

var benchWant = [][]Tag{
	[]Tag{
		mk("en"),
	},
	[]Tag{
		mk("en-AU"),
		mk("de-HK"),
		mk("nl"),
		mk("fy"),
		mk("lv"),
	},
	[]Tag{
		mk("en-AU"),
		mk("de-HK"),
		mk("nl"),
		mk("fy"),
	},
	[]Tag{
		mk("ja-Hant"),
		mk("da-HK"),
		mk("nl"),
		mk("zh-TW"),
	},
	[]Tag{
		mk("ja-Hant"),
		mk("da-HK"),
		mk("nl"),
		mk("hr"),
	},
}

func BenchmarkMatch(b *testing.B) {
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		for _, want := range benchWant {
			m.getBest(want...)
		}
	}
}

func BenchmarkMatchExact(b *testing.B) {
	want := mk("en")
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltLanguagePresent(b *testing.B) {
	want := mk("hr")
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltLanguageNotPresent(b *testing.B) {
	want := mk("nn")
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltScriptPresent(b *testing.B) {
	want := mk("zh-Hant-CN")
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltScriptNotPresent(b *testing.B) {
	want := mk("fr-Cyrl")
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchLimitedExact(b *testing.B) {
	want := []Tag{mk("he-NL"), mk("iw-NL")}
	m := newMatcher(benchHave)
	for i := 0; i < b.N; i++ {
		m.getBest(want...)
	}
}
