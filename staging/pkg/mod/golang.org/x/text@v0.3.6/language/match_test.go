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
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

var verbose = flag.Bool("verbose", false, "set to true to print the internal tables of matchers")

func TestCompliance(t *testing.T) {
	filepath.Walk("testdata", func(file string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		r, err := os.Open(file)
		if err != nil {
			t.Fatal(err)
		}
		ucd.Parse(r, func(p *ucd.Parser) {
			name := strings.Replace(path.Join(p.String(0), p.String(1)), " ", "", -1)
			if skip[name] {
				return
			}
			t.Run(info.Name()+"/"+name, func(t *testing.T) {
				supported := makeTagList(p.String(0))
				desired := makeTagList(p.String(1))
				gotCombined, index, conf := NewMatcher(supported).Match(desired...)

				gotMatch := supported[index]
				wantMatch := Raw.Make(p.String(2)) // wantMatch may be null
				if gotMatch != wantMatch {
					t.Fatalf("match: got %q; want %q (%v)", gotMatch, wantMatch, conf)
				}
				if tag := strings.TrimSpace(p.String(3)); tag != "" {
					wantCombined := Raw.MustParse(tag)
					if err == nil && gotCombined != wantCombined {
						t.Errorf("combined: got %q; want %q (%v)", gotCombined, wantCombined, conf)
					}
				}
			})
		})
		return nil
	})
}

var skip = map[string]bool{
	// TODO: bugs
	// Honor the wildcard match. This may only be useful to select non-exact
	// stuff.
	"mul,af/nl": true, // match: got "af"; want "mul"

	// TODO: include other extensions.
	// combined: got "en-GB-u-ca-buddhist-nu-arab"; want "en-GB-fonipa-t-m0-iso-i0-pinyin-u-ca-buddhist-nu-arab"
	"und,en-GB-u-sd-gbsct/en-fonipa-u-nu-Arab-ca-buddhist-t-m0-iso-i0-pinyin": true,

	// Inconsistencies with Mark Davis' implementation where it is not clear
	// which is better.

	// Inconsistencies in combined. I think the Go approach is more appropriate.
	// We could use -u-rg- as alternative.
	"und,fr/fr-BE-fonipa":              true, // combined: got "fr"; want "fr-BE-fonipa"
	"und,fr-CA/fr-BE-fonipa":           true, // combined: got "fr-CA"; want "fr-BE-fonipa"
	"und,fr-fonupa/fr-BE-fonipa":       true, // combined: got "fr-fonupa"; want "fr-BE-fonipa"
	"und,no/nn-BE-fonipa":              true, // combined: got "no"; want "no-BE-fonipa"
	"50,und,fr-CA-fonupa/fr-BE-fonipa": true, // combined: got "fr-CA-fonupa"; want "fr-BE-fonipa"

	// The initial number is a threshold. As we don't use scoring, we will not
	// implement this.
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
		tags = append(tags, mk(strings.TrimSpace(s)))
	}
	return tags
}

func TestMatchStrings(t *testing.T) {
	testCases := []struct {
		supported string
		desired   string // strings separted by |
		tag       string
		index     int
	}{{
		supported: "en",
		desired:   "",
		tag:       "en",
		index:     0,
	}, {
		supported: "en",
		desired:   "nl",
		tag:       "en",
		index:     0,
	}, {
		supported: "en,nl",
		desired:   "nl",
		tag:       "nl",
		index:     1,
	}, {
		supported: "en,nl",
		desired:   "nl|en",
		tag:       "nl",
		index:     1,
	}, {
		supported: "en-GB,nl",
		desired:   "en ; q=0.1,nl",
		tag:       "nl",
		index:     1,
	}, {
		supported: "en-GB,nl",
		desired:   "en;q=0.005 | dk; q=0.1,nl ",
		tag:       "en-GB",
		index:     0,
	}, {
		// do not match faulty tags with und
		supported: "en,und",
		desired:   "|en",
		tag:       "en",
		index:     0,
	}}
	for _, tc := range testCases {
		t.Run(path.Join(tc.supported, tc.desired), func(t *testing.T) {
			m := NewMatcher(makeTagList(tc.supported))
			tag, index := MatchStrings(m, strings.Split(tc.desired, "|")...)
			if tag.String() != tc.tag || index != tc.index {
				t.Errorf("got %v, %d; want %v, %d", tag, index, tc.tag, tc.index)
			}
		})
	}
}

func TestRegionGroups(t *testing.T) {
	testCases := []struct {
		a, b     string
		distance uint8
	}{
		{"zh-TW", "zh-HK", 5},
		{"zh-MO", "zh-HK", 4},
		{"es-ES", "es-AR", 5},
		{"es-ES", "es", 4},
		{"es-419", "es-MX", 4},
		{"es-AR", "es-MX", 4},
		{"es-ES", "es-MX", 5},
		{"es-PT", "es-MX", 5},
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
		d, _ := regionGroupDist(a.region(), b.region(), aScript.scriptID, a.lang())
		if d != tc.distance {
			t.Errorf("got %q; want %q", d, tc.distance)
		}
	}
}

func TestIsParadigmLocale(t *testing.T) {
	testCases := map[string]bool{
		"en-US":  true,
		"en-GB":  true,
		"en-VI":  false,
		"es-GB":  false,
		"es-ES":  true,
		"es-419": true,
	}
	for str, want := range testCases {
		tt := Make(str)
		tag := tt.tag()
		got := isParadigmLocale(tag.LangID, tag.RegionID)
		if got != want {
			t.Errorf("isPL(%q) = %v; want %v", str, got, want)
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
	fmt.Fprint(w, "haveTag: ")
	for _, h := range h.haveTags {
		fmt.Fprintf(w, "%v, ", h)
	}
	return w.String()
}

func (t haveTag) String() string {
	return fmt.Sprintf("%v:%d:%v:%v-%v|%v", t.tag, t.index, t.conf, t.maxRegion, t.maxScript, t.altScript)
}

func TestIssue43834(t *testing.T) {
	matcher := NewMatcher([]Tag{English})

	// ZZ is the largest region code and should not cause overflow.
	desired, _, err := ParseAcceptLanguage("en-ZZ")
	if err != nil {
		t.Error(err)
	}
	_, i, _ := matcher.Match(desired...)
	if i != 0 {
		t.Errorf("got %v; want 0", i)
	}
}

func TestBestMatchAlloc(t *testing.T) {
	m := NewMatcher(makeTagList("en sr nl"))
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
