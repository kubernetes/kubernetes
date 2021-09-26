// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build icu
// +build icu

package cases

import (
	"path"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
)

func TestICUConformance(t *testing.T) {
	// Build test set.
	input := []string{
		"a.a a_a",
		"a\u05d0a",
		"\u05d0'a",
		"a\u03084a",
		"a\u0308a",
		"a3\u30a3a",
		"a\u303aa",
		"a_\u303a_a",
		"1_a..a",
		"1_a.a",
		"a..a.",
		"a--a-",
		"a-a-",
		"a\u200ba",
		"a\u200b\u200ba",
		"a\u00ad\u00ada", // Format
		"a\u00ada",
		"a''a", // SingleQuote
		"a'a",
		"a::a", // MidLetter
		"a:a",
		"a..a", // MidNumLet
		"a.a",
		"a;;a", // MidNum
		"a;a",
		"a__a", // ExtendNumlet
		"a_a",
		"ΟΣ''a",
	}
	add := func(x interface{}) {
		switch v := x.(type) {
		case string:
			input = append(input, v)
		case []string:
			for _, s := range v {
				input = append(input, s)
			}
		}
	}
	for _, tc := range testCases {
		add(tc.src)
		add(tc.lower)
		add(tc.upper)
		add(tc.title)
	}
	for _, tc := range bufferTests {
		add(tc.src)
	}
	for _, tc := range breakTest {
		add(strings.Replace(tc, "|", "", -1))
	}
	for _, tc := range foldTestCases {
		add(tc)
	}

	// Compare ICU to Go.
	for _, c := range []string{"lower", "upper", "title", "fold"} {
		for _, tag := range []string{
			"und", "af", "az", "el", "lt", "nl", "tr",
		} {
			for _, s := range input {
				if exclude(c, tag, s) {
					continue
				}
				testtext.Run(t, path.Join(c, tag, s), func(t *testing.T) {
					want := doICU(tag, c, s)
					got := doGo(tag, c, s)
					if norm.NFC.String(got) != norm.NFC.String(want) {
						t.Errorf("\n    in %[3]q (%+[3]q)\n   got %[1]q (%+[1]q)\n  want %[2]q (%+[2]q)", got, want, s)
					}
				})
			}
		}
	}
}

// exclude indicates if a string should be excluded from testing.
func exclude(cm, tag, s string) bool {
	list := []struct{ cm, tags, pattern string }{
		// TODO: Go does not handle certain esoteric breaks correctly. This will be
		// fixed once we have a real word break iterator. Alternatively, it
		// seems like we're not too far off from making it work, so we could
		// fix these last steps. But first verify that using a separate word
		// breaker does not hurt performance.
		{"title", "af nl", "a''a"},
		{"", "", "א'a"},

		// All the exclusions below seem to be issues with the ICU
		// implementation (at version 57) and thus are not marked as TODO.

		// ICU does not handle leading apostrophe for Dutch and
		// Afrikaans correctly. See https://unicode.org/cldr/trac/ticket/7078.
		{"title", "af nl", "'n"},
		{"title", "af nl", "'N"},

		// Go terminates the final sigma check after a fixed number of
		// ignorables have been found. This ensures that the algorithm can make
		// progress in a streaming scenario.
		{"lower title", "", "\u039f\u03a3...............................a"},
		// This also applies to upper in Greek.
		// NOTE: we could fix the following two cases by adding state to elUpper
		// and aztrLower. However, considering a modifier to not belong to the
		// preceding letter after the maximum modifiers count is reached is
		// consistent with the behavior of unicode/norm.
		{"upper", "el", "\u03bf" + strings.Repeat("\u0321", 29) + "\u0313"},
		{"lower", "az tr lt", "I" + strings.Repeat("\u0321", 30) + "\u0307\u0300"},
		{"upper", "lt", "i" + strings.Repeat("\u0321", 30) + "\u0307\u0300"},
		{"lower", "lt", "I" + strings.Repeat("\u0321", 30) + "\u0300"},

		// ICU title case seems to erroneously removes \u0307 from an upper case
		// I unconditionally, instead of only when lowercasing. The ICU
		// transform algorithm transforms these cases consistently with our
		// implementation.
		{"title", "az tr", "\u0307"},

		// The spec says to remove \u0307 after Soft-Dotted characters. ICU
		// transforms conform but ucasemap_utf8ToUpper does not.
		{"upper title", "lt", "i\u0307"},
		{"upper title", "lt", "i" + strings.Repeat("\u0321", 29) + "\u0307\u0300"},

		// Both Unicode and CLDR prescribe an extra explicit dot above after a
		// Soft_Dotted character if there are other modifiers.
		// ucasemap_utf8ToUpper does not do this; ICU transforms do.
		// The issue with ucasemap_utf8ToUpper seems to be that it does not
		// consider the modifiers that are part of composition in the evaluation
		// of More_Above. For instance, according to the More_Above rule for lt,
		// a dotted capital I (U+0130) becomes i\u0307\u0307 (an small i with
		// two additional dots). This seems odd, but is correct. ICU is
		// definitely not correct as it produces different results for different
		// normal forms. For instance, for an İ:
		//    \u0130  (NFC) -> i\u0307         (incorrect)
		//    I\u0307 (NFD) -> i\u0307\u0307   (correct)
		// We could argue that we should not add a \u0307 if there already is
		// one, but this may be hard to get correct and is not conform the
		// standard.
		{"lower title", "lt", "\u0130"},
		{"lower title", "lt", "\u00cf"},

		// We are conform ICU ucasemap_utf8ToUpper if we remove support for
		// elUpper. However, this is clearly not conform the spec. Moreover, the
		// ICU transforms _do_ implement this transform and produces results
		// consistent with our implementation. Note that we still prefer to use
		// ucasemap_utf8ToUpper instead of transforms as the latter have
		// inconsistencies in the word breaking algorithm.
		{"upper", "el", "\u0386"}, // GREEK CAPITAL LETTER ALPHA WITH TONOS
		{"upper", "el", "\u0389"}, // GREEK CAPITAL LETTER ETA WITH TONOS
		{"upper", "el", "\u038A"}, // GREEK CAPITAL LETTER IOTA WITH TONOS

		{"upper", "el", "\u0391"}, // GREEK CAPITAL LETTER ALPHA
		{"upper", "el", "\u0397"}, // GREEK CAPITAL LETTER ETA
		{"upper", "el", "\u0399"}, // GREEK CAPITAL LETTER IOTA

		{"upper", "el", "\u03AC"}, // GREEK SMALL LETTER ALPHA WITH TONOS
		{"upper", "el", "\u03AE"}, // GREEK SMALL LETTER ALPHA WITH ETA
		{"upper", "el", "\u03AF"}, // GREEK SMALL LETTER ALPHA WITH IOTA

		{"upper", "el", "\u03B1"}, // GREEK SMALL LETTER ALPHA
		{"upper", "el", "\u03B7"}, // GREEK SMALL LETTER ETA
		{"upper", "el", "\u03B9"}, // GREEK SMALL LETTER IOTA
	}
	for _, x := range list {
		if x.cm != "" && strings.Index(x.cm, cm) == -1 {
			continue
		}
		if x.tags != "" && strings.Index(x.tags, tag) == -1 {
			continue
		}
		if strings.Index(s, x.pattern) != -1 {
			return true
		}
	}
	return false
}

func doGo(tag, caser, input string) string {
	var c Caser
	t := language.MustParse(tag)
	switch caser {
	case "lower":
		c = Lower(t)
	case "upper":
		c = Upper(t)
	case "title":
		c = Title(t)
	case "fold":
		c = Fold()
	}
	return c.String(input)
}
