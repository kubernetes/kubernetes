// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compact

import (
	"strings"
	"testing"

	"golang.org/x/text/internal/language"
)

var errSyntax = language.ErrSyntax

type parseTest struct {
	i                    int // the index of this test
	in                   string
	lang, script, region string
	variants, ext        string
	extList              []string // only used when more than one extension is present
	invalid              bool
	rewrite              bool // special rewrite not handled by parseTag
	changed              bool // string needed to be reformatted
}

func parseTests() []parseTest {
	tests := []parseTest{
		{in: "root", lang: "und"},
		{in: "und", lang: "und"},
		{in: "en", lang: "en"},

		{in: "en-US-u-va-posix", lang: "en", region: "US", ext: "u-va-posix"},
		{in: "ca-ES-valencia", lang: "ca", region: "ES", variants: "valencia"},
		{in: "en-US-u-rg-gbzzzz", lang: "en", region: "US", ext: "u-rg-gbzzzz"},

		{in: "xy", lang: "und", invalid: true},
		{in: "en-ZY", lang: "en", invalid: true},
		{in: "gsw", lang: "gsw"},
		{in: "sr_Latn", lang: "sr", script: "Latn"},
		{in: "af-Arab", lang: "af", script: "Arab"},
		{in: "nl-BE", lang: "nl", region: "BE"},
		{in: "es-419", lang: "es", region: "419"},
		{in: "und-001", lang: "und", region: "001"},
		{in: "de-latn-be", lang: "de", script: "Latn", region: "BE"},
		// Variants
		{in: "de-1901", lang: "de", variants: "1901"},
		// Accept with unsuppressed script.
		{in: "de-Latn-1901", lang: "de", script: "Latn", variants: "1901"},
		// Specialized.
		{in: "sl-rozaj", lang: "sl", variants: "rozaj"},
		{in: "sl-rozaj-lipaw", lang: "sl", variants: "rozaj-lipaw"},
		{in: "sl-rozaj-biske", lang: "sl", variants: "rozaj-biske"},
		{in: "sl-rozaj-biske-1994", lang: "sl", variants: "rozaj-biske-1994"},
		{in: "sl-rozaj-1994", lang: "sl", variants: "rozaj-1994"},
		// Maximum number of variants while adhering to prefix rules.
		{in: "sl-rozaj-biske-1994-alalc97-fonipa-fonupa-fonxsamp", lang: "sl", variants: "rozaj-biske-1994-alalc97-fonipa-fonupa-fonxsamp"},

		// Sorting.
		{in: "sl-1994-biske-rozaj", lang: "sl", variants: "rozaj-biske-1994", changed: true},
		{in: "sl-rozaj-biske-1994-alalc97-fonupa-fonipa-fonxsamp", lang: "sl", variants: "rozaj-biske-1994-alalc97-fonipa-fonupa-fonxsamp", changed: true},
		{in: "nl-fonxsamp-alalc97-fonipa-fonupa", lang: "nl", variants: "alalc97-fonipa-fonupa-fonxsamp", changed: true},

		// Duplicates variants are removed, but not an error.
		{in: "nl-fonupa-fonupa", lang: "nl", variants: "fonupa"},

		// Variants that do not have correct prefixes. We still accept these.
		{in: "de-Cyrl-1901", lang: "de", script: "Cyrl", variants: "1901"},
		{in: "sl-rozaj-lipaw-1994", lang: "sl", variants: "rozaj-lipaw-1994"},
		{in: "sl-1994-biske-rozaj-1994-biske-rozaj", lang: "sl", variants: "rozaj-biske-1994", changed: true},
		{in: "de-Cyrl-1901", lang: "de", script: "Cyrl", variants: "1901"},

		// Invalid variant.
		{in: "de-1902", lang: "de", variants: "", invalid: true},

		{in: "EN_CYRL", lang: "en", script: "Cyrl"},
		// private use and extensions
		{in: "x-a-b-c-d", ext: "x-a-b-c-d"},
		{in: "x_A.-B-C_D", ext: "x-b-c-d", invalid: true, changed: true},
		{in: "x-aa-bbbb-cccccccc-d", ext: "x-aa-bbbb-cccccccc-d"},
		{in: "en-c_cc-b-bbb-a-aaa", lang: "en", changed: true, extList: []string{"a-aaa", "b-bbb", "c-cc"}},
		{in: "en-x_cc-b-bbb-a-aaa", lang: "en", ext: "x-cc-b-bbb-a-aaa", changed: true},
		{in: "en-c_cc-b-bbb-a-aaa-x-x", lang: "en", changed: true, extList: []string{"a-aaa", "b-bbb", "c-cc", "x-x"}},
		{in: "en-v-c", lang: "en", ext: "", invalid: true},
		{in: "en-v-abcdefghi", lang: "en", ext: "", invalid: true},
		{in: "en-v-abc-x", lang: "en", ext: "v-abc", invalid: true},
		{in: "en-v-abc-x-", lang: "en", ext: "v-abc", invalid: true},
		{in: "en-v-abc-w-x-xx", lang: "en", extList: []string{"v-abc", "x-xx"}, invalid: true, changed: true},
		{in: "en-v-abc-w-y-yx", lang: "en", extList: []string{"v-abc", "y-yx"}, invalid: true, changed: true},
		{in: "en-v-c-abc", lang: "en", ext: "c-abc", invalid: true, changed: true},
		{in: "en-v-w-abc", lang: "en", ext: "w-abc", invalid: true, changed: true},
		{in: "en-v-x-abc", lang: "en", ext: "x-abc", invalid: true, changed: true},
		{in: "en-v-x-a", lang: "en", ext: "x-a", invalid: true, changed: true},
		{in: "en-9-aa-0-aa-z-bb-x-a", lang: "en", extList: []string{"0-aa", "9-aa", "z-bb", "x-a"}, changed: true},
		{in: "en-u-c", lang: "en", ext: "", invalid: true},
		{in: "en-u-co-phonebk", lang: "en", ext: "u-co-phonebk"},
		{in: "en-u-co-phonebk-ca", lang: "en", ext: "u-co-phonebk", invalid: true},
		{in: "en-u-nu-arabic-co-phonebk-ca", lang: "en", ext: "u-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-x", lang: "en", ext: "u-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-s", lang: "en", ext: "u-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-a12345678", lang: "en", ext: "u-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-co-phonebook", lang: "en", ext: "", invalid: true},
		{in: "en-u-co-phonebook-cu-xau", lang: "en", ext: "u-cu-xau", invalid: true, changed: true},
		{in: "en-Cyrl-u-co-phonebk", lang: "en", script: "Cyrl", ext: "u-co-phonebk"},
		{in: "en-US-u-co-phonebk", lang: "en", region: "US", ext: "u-co-phonebk"},
		{in: "en-US-u-co-phonebk-cu-xau", lang: "en", region: "US", ext: "u-co-phonebk-cu-xau"},
		{in: "en-scotland-u-co-phonebk", lang: "en", variants: "scotland", ext: "u-co-phonebk"},
		{in: "en-u-cu-xua-co-phonebk", lang: "en", ext: "u-co-phonebk-cu-xua", changed: true},
		{in: "en-u-def-abc-cu-xua-co-phonebk", lang: "en", ext: "u-abc-def-co-phonebk-cu-xua", changed: true},
		{in: "en-u-def-abc", lang: "en", ext: "u-abc-def", changed: true},
		{in: "en-u-cu-xua-co-phonebk-a-cd", lang: "en", extList: []string{"a-cd", "u-co-phonebk-cu-xua"}, changed: true},
		// Invalid "u" extension. Drop invalid parts.
		{in: "en-u-cu-co-phonebk", lang: "en", extList: []string{"u-co-phonebk"}, invalid: true, changed: true},
		{in: "en-u-cu-xau-co", lang: "en", extList: []string{"u-cu-xau"}, invalid: true},
		// We allow duplicate keys as the LDML spec does not explicitly prohibit it.
		// TODO: Consider eliminating duplicates and returning an error.
		{in: "en-u-cu-xau-co-phonebk-cu-xau", lang: "en", ext: "u-co-phonebk-cu-xau", changed: true},
		{in: "en-t-en-Cyrl-NL-fonipa", lang: "en", ext: "t-en-cyrl-nl-fonipa", changed: true},
		{in: "en-t-en-Cyrl-NL-fonipa-t0-abc-def", lang: "en", ext: "t-en-cyrl-nl-fonipa-t0-abc-def", changed: true},
		{in: "en-t-t0-abcd", lang: "en", ext: "t-t0-abcd"},
		// Not necessary to have changed here.
		{in: "en-t-nl-abcd", lang: "en", ext: "t-nl", invalid: true},
		{in: "en-t-nl-latn", lang: "en", ext: "t-nl-latn"},
		{in: "en-t-t0-abcd-x-a", lang: "en", extList: []string{"t-t0-abcd", "x-a"}},
		// invalid
		{in: "", lang: "und", invalid: true},
		{in: "-", lang: "und", invalid: true},
		{in: "x", lang: "und", invalid: true},
		{in: "x-", lang: "und", invalid: true},
		{in: "x--", lang: "und", invalid: true},
		{in: "a-a-b-c-d", lang: "und", invalid: true},
		{in: "en-", lang: "en", invalid: true},
		{in: "enne-", lang: "und", invalid: true},
		{in: "en.", lang: "und", invalid: true},
		{in: "en.-latn", lang: "und", invalid: true},
		{in: "en.-en", lang: "en", invalid: true},
		{in: "x-a-tooManyChars-c-d", ext: "x-a-c-d", invalid: true, changed: true},
		{in: "a-tooManyChars-c-d", lang: "und", invalid: true},
		// TODO: check key-value validity
		// { in: "en-u-cu-xd", lang: "en", ext: "u-cu-xd", invalid: true },
		{in: "en-t-abcd", lang: "en", invalid: true},
		{in: "en-Latn-US-en", lang: "en", script: "Latn", region: "US", invalid: true},
		// rewrites (more tests in TestGrandfathered)
		{in: "zh-min-nan", lang: "nan"},
		{in: "zh-yue", lang: "yue"},
		{in: "zh-xiang", lang: "hsn", rewrite: true},
		{in: "zh-guoyu", lang: "cmn", rewrite: true},
		{in: "iw", lang: "iw"},
		{in: "sgn-BE-FR", lang: "sfb", rewrite: true},
		{in: "i-klingon", lang: "tlh", rewrite: true},
	}
	for i, tt := range tests {
		tests[i].i = i
		if tt.extList != nil {
			tests[i].ext = strings.Join(tt.extList, "-")
		}
		if tt.ext != "" && tt.extList == nil {
			tests[i].extList = []string{tt.ext}
		}
	}
	return tests
}

// partChecks runs checks for each part by calling the function returned by f.
func partChecks(t *testing.T, f func(*parseTest) (Tag, bool)) {
	for i, tt := range parseTests() {
		tag, skip := f(&tt)
		if skip {
			continue
		}
		if l, _ := language.ParseBase(tt.lang); l != tag.Tag().LangID {
			t.Errorf("%d: lang was %q; want %q", i, tag.Tag().LangID, l)
		}
		if sc, _ := language.ParseScript(tt.script); sc != tag.Tag().ScriptID {
			t.Errorf("%d: script was %q; want %q", i, tag.Tag().ScriptID, sc)
		}
		if r, _ := language.ParseRegion(tt.region); r != tag.Tag().RegionID {
			t.Errorf("%d: region was %q; want %q", i, tag.Tag().RegionID, r)
		}
		v := tag.Tag().Variants()
		if v != "" {
			v = v[1:]
		}
		if v != tt.variants {
			t.Errorf("%d: variants was %q; want %q", i, v, tt.variants)
		}
		if e := strings.Join(tag.Tag().Extensions(), "-"); e != tt.ext {
			t.Errorf("%d: extensions were %q; want %q", i, e, tt.ext)
		}
	}
}

func mk(s string) Tag {
	tag, _ := language.Parse(s)
	return Make(tag)
}
