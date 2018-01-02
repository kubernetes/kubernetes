// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

var verbose = flag.Bool("verbose", false, "set to true to print the internal tables of matchers")

func TestCLDRCompliance(t *testing.T) {
	r, err := os.Open("testdata/localeMatcherTest.txt")
	if err != nil {
		t.Fatal(err)
	}
	ucd.Parse(r, func(p *ucd.Parser) {
		name := strings.Replace(path.Join(p.String(0), p.String(1)), " ", "", -1)
		if skip[name] {
			return
		}
		t.Run(name, func(t *testing.T) {
			supported := makeTagList(p.String(0))
			desired := makeTagList(p.String(1))
			gotCombined, index, _ := NewMatcher(supported).Match(desired...)

			gotMatch := supported[index]
			wantMatch := Make(p.String(2))
			if gotMatch != wantMatch {
				t.Fatalf("match: got %q; want %q", gotMatch, wantMatch)
			}
			wantCombined, err := Parse(p.String(3))
			if err == nil && gotCombined != wantCombined {
				t.Errorf("combined: got %q; want %q", gotCombined, wantCombined)
			}
		})
	})
}

var skip = map[string]bool{
	// TODO: bugs
	// und-<region> is not expanded to the appropriate language.
	"en-Hant-TW,und-TW/zh-Hant": true, // match: got "en-Hant-TW"; want "und-TW"
	"en-Hant-TW,und-TW/zh":      true, // match: got "en-Hant-TW"; want "und-TW"
	// Honor the wildcard match. This may only be useful to select non-exact
	// stuff.
	"mul,af/nl": true, // match: got "af"; want "mul"

	// TODO: include other extensions.
	// combined: got "en-GB-u-ca-buddhist-nu-arab"; want "en-GB-fonipa-t-m0-iso-i0-pinyin-u-ca-buddhist-nu-arab"
	"und,en-GB-u-sd-gbsct/en-fonipa-u-nu-Arab-ca-buddhist-t-m0-iso-i0-pinyin": true,

	// Inconsistencies with Mark Davis' implementation where it is not clear
	// which is better.

	// Go prefers exact matches over less exact preferred ones.
	// Preferring desired ones might be better.
	"en,de,fr,ja/de-CH,fr":              true, // match: got "fr"; want "de"
	"en-GB,en,de,fr,ja/de-CH,fr":        true, // match: got "fr"; want "de"
	"pt-PT,pt-BR,es,es-419/pt-US,pt-PT": true, // match: got "pt-PT"; want "pt-BR"
	"pt-PT,pt,es,es-419/pt-US,pt-PT,pt": true, // match: got "pt-PT"; want "pt"
	"en,sv/en-GB,sv":                    true, // match: got "sv"; want "en"
	"en-NZ,en-IT/en-US":                 true, // match: got "en-IT"; want "en-NZ"

	// Inconsistencies in combined. I think the Go approach is more appropriate.
	// We could use -u-rg- and -u-va- as alternative.
	"und,fr/fr-BE-fonipa":              true, // combined: got "fr"; want "fr-BE-fonipa"
	"und,fr-CA/fr-BE-fonipa":           true, // combined: got "fr-CA"; want "fr-BE-fonipa"
	"und,fr-fonupa/fr-BE-fonipa":       true, // combined: got "fr-fonupa"; want "fr-BE-fonipa"
	"und,no/nn-BE-fonipa":              true, // combined: got "no"; want "no-BE-fonipa"
	"50,und,fr-CA-fonupa/fr-BE-fonipa": true, // combined: got "fr-CA-fonupa"; want "fr-BE-fonipa"

	// Spec says prefer primary locales. But what is the benefit? Shouldn't
	// the developer just not specify the primary locale first in the list?
	// TODO: consider adding a SortByPreferredLocale function to ensure tags
	// are ordered such that the preferred locale rule is observed.
	// TODO: most of these cases are solved by getting rid of the region
	// distance tie-breaker rule (see comments there).
	"und,es,es-MA,es-MX,es-419/es-EA": true, // match: got "es-MA"; want "es"
	"und,es-MA,es,es-419,es-MX/es-EA": true, // match: got "es-MA"; want "es"
	"und,en,en-GU,en-IN,en-GB/en-ZA":  true, // match: got "en-IN"; want "en-GB"
	"und,en,en-GU,en-IN,en-GB/en-VI":  true, // match: got "en-GU"; want "en"
	"und,en-GU,en,en-GB,en-IN/en-VI":  true, // match: got "en-GU"; want "en"

	// Falling back to the default seems more appropriate than falling back
	// on a language with the same script.
	"50,und,fr-Cyrl-CA-fonupa/fr-BE-fonipa": true,
	// match: got "und"; want "fr-Cyrl-CA-fonupa"
	// combined: got "und"; want "fr-Cyrl-BE-fonipa"

	// Other interesting cases to test:
	// - Should same language or same script have the preference if there is
	//   usually no understanding of the other script?
	// - More specific region in desired may replace enclosing supported.
}

func makeTagList(s string) (tags []Tag) {
	for _, s := range strings.Split(s, ",") {
		tags = append(tags, Make(strings.TrimSpace(s)))
	}
	return tags
}

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

func TestRegionGroups(t *testing.T) {
	testCases := []struct {
		a, b     string
		distance uint8
	}{
		{"zh-TW", "zh-HK", 5},
		{"zh-MO", "zh-HK", 4},
	}
	for _, tc := range testCases {
		a := MustParse(tc.a)
		aScript, _ := a.Script()
		b := MustParse(tc.b)
		bScript, _ := b.Script()

		if aScript != bScript {
			t.Errorf("scripts differ: %q vs %q", aScript, bScript)
			continue
		}
		d := regionGroupDist(a.region, b.region, aScript.scriptID, a.lang)
		if d != tc.distance {
			t.Errorf("got %q; want %q", d, tc.distance)
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
		testtext.Run(t, tt.a+"/"+tt.b, func(t *testing.T) {
			ra, _ := getRegionID([]byte(tt.a))
			rb, _ := getRegionID([]byte(tt.b))
			if d := regionDistance(ra, rb); d != tt.d {
				t.Errorf("%d: d(%s, %s) = %v; want %v", i, tt.a, tt.b, d, tt.d)
			}
		})
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

func parseSupported(list string) (out []Tag) {
	for _, s := range strings.Split(list, ",") {
		out = append(out, mk(strings.TrimSpace(s)))
	}
	return out
}

// The test set for TestBestMatch is defined in data_test.go.
func TestBestMatch(t *testing.T) {
	for _, tt := range matchTests {
		supported := parseSupported(tt.supported)
		m := newMatcher(supported, nil)
		if *verbose {
			fmt.Printf("%s:\n%v\n", tt.comment, m)
		}
		for _, tm := range tt.test {
			t.Run(path.Join(tt.comment, tt.supported, tm.desired), func(t *testing.T) {
				tag, _, conf := m.Match(parseSupported(tm.desired)...)
				if tag.String() != tm.match {
					t.Errorf("find %s in %q: have %s; want %s (%v)", tm.desired, tt.supported, tag, tm.match, conf)
				}
			})

		}
	}
}

func TestBestMatchAlloc(t *testing.T) {
	m := NewMatcher(parseSupported("en sr nl"))
	// Go allocates when creating a list of tags from a single tag!
	list := []Tag{English}
	avg := testtext.AllocsPerRun(1, func() {
		m.Match(list...)
	})
	if avg > 0 {
		t.Errorf("got %f; want 0", avg)
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
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		for _, want := range benchWant {
			m.getBest(want...)
		}
	}
}

func BenchmarkMatchExact(b *testing.B) {
	want := mk("en")
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltLanguagePresent(b *testing.B) {
	want := mk("hr")
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltLanguageNotPresent(b *testing.B) {
	want := mk("nn")
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltScriptPresent(b *testing.B) {
	want := mk("zh-Hant-CN")
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchAltScriptNotPresent(b *testing.B) {
	want := mk("fr-Cyrl")
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want)
	}
}

func BenchmarkMatchLimitedExact(b *testing.B) {
	want := []Tag{mk("he-NL"), mk("iw-NL")}
	m := newMatcher(benchHave, nil)
	for i := 0; i < b.N; i++ {
		m.getBest(want...)
	}
}
