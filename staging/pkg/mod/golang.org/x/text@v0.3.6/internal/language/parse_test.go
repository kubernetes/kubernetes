// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"bytes"
	"strings"
	"testing"

	"golang.org/x/text/internal/tag"
)

type scanTest struct {
	ok  bool // true if scanning does not result in an error
	in  string
	tok []string // the expected tokens
}

var tests = []scanTest{
	{true, "", []string{}},
	{true, "1", []string{"1"}},
	{true, "en", []string{"en"}},
	{true, "root", []string{"root"}},
	{true, "maxchars", []string{"maxchars"}},
	{false, "bad/", []string{}},
	{false, "morethan8", []string{}},
	{false, "-", []string{}},
	{false, "----", []string{}},
	{false, "_", []string{}},
	{true, "en-US", []string{"en", "US"}},
	{true, "en_US", []string{"en", "US"}},
	{false, "en-US-", []string{"en", "US"}},
	{false, "en-US--", []string{"en", "US"}},
	{false, "en-US---", []string{"en", "US"}},
	{false, "en--US", []string{"en", "US"}},
	{false, "-en-US", []string{"en", "US"}},
	{false, "-en--US-", []string{"en", "US"}},
	{false, "-en--US-", []string{"en", "US"}},
	{false, "en-.-US", []string{"en", "US"}},
	{false, ".-en--US-.", []string{"en", "US"}},
	{false, "en-u.-US", []string{"en", "US"}},
	{true, "en-u1-US", []string{"en", "u1", "US"}},
	{true, "maxchar1_maxchar2-maxchar3", []string{"maxchar1", "maxchar2", "maxchar3"}},
	{false, "moreThan8-moreThan8-e", []string{"e"}},
}

func TestScan(t *testing.T) {
	for i, tt := range tests {
		scan := makeScannerString(tt.in)
		for j := 0; !scan.done; j++ {
			if j >= len(tt.tok) {
				t.Errorf("%d: extra token %q", i, scan.token)
			} else if tag.Compare(tt.tok[j], scan.token) != 0 {
				t.Errorf("%d: token %d: found %q; want %q", i, j, scan.token, tt.tok[j])
				break
			}
			scan.scan()
		}
		if s := strings.Join(tt.tok, "-"); tag.Compare(s, bytes.Replace(scan.b, b("_"), b("-"), -1)) != 0 {
			t.Errorf("%d: input: found %q; want %q", i, scan.b, s)
		}
		if (scan.err == nil) != tt.ok {
			t.Errorf("%d: ok: found %v; want %v", i, scan.err == nil, tt.ok)
		}
	}
}

func TestAcceptMinSize(t *testing.T) {
	for i, tt := range tests {
		// count number of successive tokens with a minimum size.
		for sz := 1; sz <= 8; sz++ {
			scan := makeScannerString(tt.in)
			scan.end, scan.next = 0, 0
			end := scan.acceptMinSize(sz)
			n := 0
			for i := 0; i < len(tt.tok) && len(tt.tok[i]) >= sz; i++ {
				n += len(tt.tok[i])
				if i > 0 {
					n++
				}
			}
			if end != n {
				t.Errorf("%d:%d: found len %d; want %d", i, sz, end, n)
			}
		}
	}
}

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
		{in: "en-u-co-phonebk-ca", lang: "en", ext: "u-ca-co-phonebk", changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca", lang: "en", ext: "u-ca-co-phonebk-nu-arabic", changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-x", lang: "en", ext: "u-ca-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-s", lang: "en", ext: "u-ca-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-nu-arabic-co-phonebk-ca-a12345678", lang: "en", ext: "u-ca-co-phonebk-nu-arabic", invalid: true, changed: true},
		{in: "en-u-co-phonebook", lang: "en", ext: "u-co", invalid: true},
		{in: "en-u-co-phonebook-cu-xau", lang: "en", ext: "u-co-cu-xau", invalid: true, changed: true},
		{in: "en-Cyrl-u-co-phonebk", lang: "en", script: "Cyrl", ext: "u-co-phonebk"},
		{in: "en-US-u-co-phonebk", lang: "en", region: "US", ext: "u-co-phonebk"},
		{in: "en-US-u-co-phonebk-cu-xau", lang: "en", region: "US", ext: "u-co-phonebk-cu-xau"},
		{in: "en-scotland-u-co-phonebk", lang: "en", variants: "scotland", ext: "u-co-phonebk"},
		{in: "en-u-cu-xua-co-phonebk", lang: "en", ext: "u-co-phonebk-cu-xua", changed: true},
		{in: "en-u-def-abc-cu-xua-co-phonebk", lang: "en", ext: "u-abc-def-co-phonebk-cu-xua", changed: true},
		{in: "en-u-def-abc", lang: "en", ext: "u-abc-def", changed: true},
		{in: "en-u-cu-xua-co-phonebk-a-cd", lang: "en", extList: []string{"a-cd", "u-co-phonebk-cu-xua"}, changed: true},
		{in: "en-u-cu-co-phonebk", lang: "en", extList: []string{"u-co-phonebk-cu"}, changed: true},
		{in: "en-u-cu-xau-co", lang: "en", extList: []string{"u-co-cu-xau"}, changed: true},
		// LDML spec is not specific about it, but remove duplicates and return an error if the values differ.
		{in: "en-u-cu-xau-co-phonebk-cu-xau", lang: "en", ext: "u-co-phonebk-cu-xau", changed: true},
		// No change as the result is a substring of the original!
		{in: "en-US-u-cu-xau-cu-eur", lang: "en", region: "US", ext: "u-cu-xau", invalid: true, changed: false},
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

func TestParseExtensions(t *testing.T) {
	for i, tt := range parseTests() {
		if tt.ext == "" || tt.rewrite {
			continue
		}
		scan := makeScannerString(tt.in)
		if len(scan.b) > 1 && scan.b[1] != '-' {
			scan.end = nextExtension(string(scan.b), 0)
			scan.next = scan.end + 1
			scan.scan()
		}
		start := scan.start
		scan.toLower(start, len(scan.b))
		parseExtensions(&scan)
		ext := string(scan.b[start:])
		if ext != tt.ext {
			t.Errorf("%d(%s): ext was %v; want %v", i, tt.in, ext, tt.ext)
		}
		if changed := !strings.HasPrefix(tt.in[start:], ext); changed != tt.changed {
			t.Errorf("%d(%s): changed was %v; want %v", i, tt.in, changed, tt.changed)
		}
	}
}

// partChecks runs checks for each part by calling the function returned by f.
func partChecks(t *testing.T, f func(*testing.T, *parseTest) (Tag, bool)) {
	for i, tt := range parseTests() {
		t.Run(tt.in, func(t *testing.T) {
			tag, skip := f(t, &tt)
			if skip {
				return
			}
			if l, _ := getLangID(b(tt.lang)); l != tag.LangID {
				t.Errorf("%d: lang was %q; want %q", i, tag.LangID, l)
			}
			if sc, _ := getScriptID(script, b(tt.script)); sc != tag.ScriptID {
				t.Errorf("%d: script was %q; want %q", i, tag.ScriptID, sc)
			}
			if r, _ := getRegionID(b(tt.region)); r != tag.RegionID {
				t.Errorf("%d: region was %q; want %q", i, tag.RegionID, r)
			}
			if tag.str == "" {
				return
			}
			p := int(tag.pVariant)
			if p < int(tag.pExt) {
				p++
			}
			if s, g := tag.str[p:tag.pExt], tt.variants; s != g {
				t.Errorf("%d: variants was %q; want %q", i, s, g)
			}
			p = int(tag.pExt)
			if p > 0 && p < len(tag.str) {
				p++
			}
			if s, g := (tag.str)[p:], tt.ext; s != g {
				t.Errorf("%d: extensions were %q; want %q", i, s, g)
			}
		})
	}
}

func TestParseTag(t *testing.T) {
	partChecks(t, func(t *testing.T, tt *parseTest) (id Tag, skip bool) {
		if strings.HasPrefix(tt.in, "x-") || tt.rewrite {
			return Tag{}, true
		}
		scan := makeScannerString(tt.in)
		id, end := parseTag(&scan)
		id.str = string(scan.b[:end])
		tt.ext = ""
		tt.extList = []string{}
		return id, false
	})
}

func TestParse(t *testing.T) {
	partChecks(t, func(t *testing.T, tt *parseTest) (id Tag, skip bool) {
		id, err := Parse(tt.in)
		ext := ""
		if id.str != "" {
			if strings.HasPrefix(id.str, "x-") {
				ext = id.str
			} else if int(id.pExt) < len(id.str) && id.pExt > 0 {
				ext = id.str[id.pExt+1:]
			}
		}
		if tag, _ := Parse(id.String()); tag.String() != id.String() {
			t.Errorf("%d:%s: reparse was %q; want %q", tt.i, tt.in, id.String(), tag.String())
		}
		if ext != tt.ext {
			t.Errorf("%d:%s: ext was %q; want %q", tt.i, tt.in, ext, tt.ext)
		}
		changed := id.str != "" && !strings.HasPrefix(tt.in, id.str)
		if changed != tt.changed {
			t.Errorf("%d:%s: changed was %v; want %v", tt.i, tt.in, changed, tt.changed)
		}
		if (err != nil) != tt.invalid {
			t.Errorf("%d:%s: invalid was %v; want %v. Error: %v", tt.i, tt.in, err != nil, tt.invalid, err)
		}
		return id, false
	})
}

func TestErrors(t *testing.T) {
	mkInvalid := func(s string) error {
		return NewValueError([]byte(s))
	}
	tests := []struct {
		in  string
		out error
	}{
		// invalid subtags.
		{"ac", mkInvalid("ac")},
		{"AC", mkInvalid("ac")},
		{"aa-Uuuu", mkInvalid("Uuuu")},
		{"aa-AB", mkInvalid("AB")},
		// ill-formed wins over invalid.
		{"ac-u", ErrSyntax},
		{"ac-u-ca", mkInvalid("ac")},
		{"ac-u-ca-co-pinyin", mkInvalid("ac")},
		{"noob", ErrSyntax},
	}
	for _, tt := range tests {
		_, err := Parse(tt.in)
		if err != tt.out {
			t.Errorf("%s: was %q; want %q", tt.in, err, tt.out)
		}
	}
}
