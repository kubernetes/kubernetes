// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"reflect"
	"testing"

	"golang.org/x/text/internal/testtext"
)

func TestTagSize(t *testing.T) {
	id := Tag{}
	typ := reflect.TypeOf(id)
	if typ.Size() > 24 {
		t.Errorf("size of Tag was %d; want 24", typ.Size())
	}
}

func TestIsRoot(t *testing.T) {
	loc := Tag{}
	if !loc.IsRoot() {
		t.Errorf("unspecified should be root.")
	}
	for i, tt := range parseTests() {
		loc, _ := Parse(tt.in)
		undef := tt.lang == "und" && tt.script == "" && tt.region == "" && tt.ext == ""
		if loc.IsRoot() != undef {
			t.Errorf("%d: was %v; want %v", i, loc.IsRoot(), undef)
		}
	}
}

func TestEquality(t *testing.T) {
	for i, tt := range parseTests() {
		s := tt.in
		tag := Make(s)
		t1 := Make(tag.String())
		if tag != t1 {
			t.Errorf("%d:%s: equality test 1 failed\n got: %#v\nwant: %#v)", i, s, t1, tag)
		}
	}
}

func TestMakeString(t *testing.T) {
	tests := []struct{ in, out string }{
		{"und", "und"},
		{"und", "und-CW"},
		{"nl", "nl-NL"},
		{"de-1901", "nl-1901"},
		{"de-1901", "de-Arab-1901"},
		{"x-a-b", "de-Arab-x-a-b"},
		{"x-a-b", "x-a-b"},
	}
	for i, tt := range tests {
		id, _ := Parse(tt.in)
		mod, _ := Parse(tt.out)
		id.setTagsFrom(mod)
		for j := 0; j < 2; j++ {
			id.RemakeString()
			if str := id.String(); str != tt.out {
				t.Errorf("%d:%d: found %s; want %s", i, j, id.String(), tt.out)
			}
		}
		// The bytes to string conversion as used in remakeString
		// occasionally measures as more than one alloc, breaking this test.
		// To alleviate this we set the number of runs to more than 1.
		if n := testtext.AllocsPerRun(8, id.RemakeString); n > 1 {
			t.Errorf("%d: # allocs got %.1f; want <= 1", i, n)
		}
	}
}

func TestMarshal(t *testing.T) {
	testCases := []string{
		// TODO: these values will change with each CLDR update. This issue
		// will be solved if we decide to fix the indexes.
		"und",
		"ca-ES-valencia",
		"ca-ES-valencia-u-va-posix",
		"ca-ES-valencia-u-co-phonebk",
		"ca-ES-valencia-u-co-phonebk-va-posix",
		"x-klingon",
		"en-US",
		"en-US-u-va-posix",
		"en",
		"en-u-co-phonebk",
		"en-001",
		"sh",
	}
	for _, tc := range testCases {
		var tag Tag
		err := tag.UnmarshalText([]byte(tc))
		if err != nil {
			t.Errorf("UnmarshalText(%q): unexpected error: %v", tc, err)
		}
		b, err := tag.MarshalText()
		if err != nil {
			t.Errorf("MarshalText(%q): unexpected error: %v", tc, err)
		}
		if got := string(b); got != tc {
			t.Errorf("%s: got %q; want %q", tc, got, tc)
		}
	}
}

func TestParseBase(t *testing.T) {
	tests := []struct {
		in  string
		out string
		ok  bool
	}{
		{"en", "en", true},
		{"EN", "en", true},
		{"nld", "nl", true},
		{"dut", "dut", true},  // bibliographic
		{"aaj", "und", false}, // unknown
		{"qaa", "qaa", true},
		{"a", "und", false},
		{"", "und", false},
		{"aaaa", "und", false},
	}
	for i, tt := range tests {
		x, err := ParseBase(tt.in)
		if x.String() != tt.out || err == nil != tt.ok {
			t.Errorf("%d:%s: was %s, %v; want %s, %v", i, tt.in, x, err == nil, tt.out, tt.ok)
		}
		if y, _, _ := Make(tt.out).Raw(); x != y {
			t.Errorf("%d:%s: tag was %s; want %s", i, tt.in, x, y)
		}
	}
}

func TestParseScript(t *testing.T) {
	tests := []struct {
		in  string
		out string
		ok  bool
	}{
		{"Latn", "Latn", true},
		{"zzzz", "Zzzz", true},
		{"zyyy", "Zyyy", true},
		{"Latm", "Zzzz", false},
		{"Zzz", "Zzzz", false},
		{"", "Zzzz", false},
		{"Zzzxx", "Zzzz", false},
	}
	for i, tt := range tests {
		x, err := ParseScript(tt.in)
		if x.String() != tt.out || err == nil != tt.ok {
			t.Errorf("%d:%s: was %s, %v; want %s, %v", i, tt.in, x, err == nil, tt.out, tt.ok)
		}
		if err == nil {
			if _, y, _ := Make("und-" + tt.out).Raw(); x != y {
				t.Errorf("%d:%s: tag was %s; want %s", i, tt.in, x, y)
			}
		}
	}
}

func TestEncodeM49(t *testing.T) {
	tests := []struct {
		m49  int
		code string
		ok   bool
	}{
		{1, "001", true},
		{840, "US", true},
		{899, "ZZ", false},
	}
	for i, tt := range tests {
		if r, err := EncodeM49(tt.m49); r.String() != tt.code || err == nil != tt.ok {
			t.Errorf("%d:%d: was %s, %v; want %s, %v", i, tt.m49, r, err == nil, tt.code, tt.ok)
		}
	}
	for i := 1; i <= 1000; i++ {
		if r, err := EncodeM49(i); err == nil && r.M49() == 0 {
			t.Errorf("%d has no error, but maps to undefined region", i)
		}
	}
}

func TestParseRegion(t *testing.T) {
	tests := []struct {
		in  string
		out string
		ok  bool
	}{
		{"001", "001", true},
		{"840", "US", true},
		{"899", "ZZ", false},
		{"USA", "US", true},
		{"US", "US", true},
		{"BC", "ZZ", false},
		{"C", "ZZ", false},
		{"CCCC", "ZZ", false},
		{"01", "ZZ", false},
	}
	for i, tt := range tests {
		r, err := ParseRegion(tt.in)
		if r.String() != tt.out || err == nil != tt.ok {
			t.Errorf("%d:%s: was %s, %v; want %s, %v", i, tt.in, r, err == nil, tt.out, tt.ok)
		}
		if err == nil {
			if _, _, y := Make("und-" + tt.out).Raw(); r != y {
				t.Errorf("%d:%s: tag was %s; want %s", i, tt.in, r, y)
			}
		}
	}
}

func TestIsCountry(t *testing.T) {
	tests := []struct {
		reg     string
		country bool
	}{
		{"US", true},
		{"001", false},
		{"958", false},
		{"419", false},
		{"203", true},
		{"020", true},
		{"900", false},
		{"999", false},
		{"QO", false},
		{"EU", false},
		{"AA", false},
		{"XK", true},
	}
	for i, tt := range tests {
		r, _ := getRegionID([]byte(tt.reg))
		if r.IsCountry() != tt.country {
			t.Errorf("%d: IsCountry(%s) was %v; want %v", i, tt.reg, r.IsCountry(), tt.country)
		}
	}
}

func TestIsGroup(t *testing.T) {
	tests := []struct {
		reg   string
		group bool
	}{
		{"US", false},
		{"001", true},
		{"958", false},
		{"419", true},
		{"203", false},
		{"020", false},
		{"900", false},
		{"999", false},
		{"QO", true},
		{"EU", true},
		{"AA", false},
		{"XK", false},
	}
	for i, tt := range tests {
		r, _ := getRegionID([]byte(tt.reg))
		if r.IsGroup() != tt.group {
			t.Errorf("%d: IsGroup(%s) was %v; want %v", i, tt.reg, r.IsGroup(), tt.group)
		}
	}
}

func TestContains(t *testing.T) {
	tests := []struct {
		enclosing, contained string
		contains             bool
	}{
		// A region contains itself.
		{"US", "US", true},
		{"001", "001", true},

		// Direct containment.
		{"001", "002", true},
		{"039", "XK", true},
		{"150", "XK", true},
		{"EU", "AT", true},
		{"QO", "AQ", true},

		// Indirect containemnt.
		{"001", "US", true},
		{"001", "419", true},
		{"001", "013", true},

		// No containment.
		{"US", "001", false},
		{"155", "EU", false},
	}
	for i, tt := range tests {
		enc, _ := getRegionID([]byte(tt.enclosing))
		con, _ := getRegionID([]byte(tt.contained))
		r := enc
		if got := r.Contains(con); got != tt.contains {
			t.Errorf("%d: %s.Contains(%s) was %v; want %v", i, tt.enclosing, tt.contained, got, tt.contains)
		}
	}
}

func TestRegionCanonicalize(t *testing.T) {
	for i, tt := range []struct{ in, out string }{
		{"UK", "GB"},
		{"TP", "TL"},
		{"QU", "EU"},
		{"SU", "SU"},
		{"VD", "VN"},
		{"DD", "DE"},
	} {
		r := MustParseRegion(tt.in)
		want := MustParseRegion(tt.out)
		if got := r.Canonicalize(); got != want {
			t.Errorf("%d: got %v; want %v", i, got, want)
		}
	}
}

func TestRegionTLD(t *testing.T) {
	for _, tt := range []struct {
		in, out string
		ok      bool
	}{
		{"EH", "EH", true},
		{"FR", "FR", true},
		{"TL", "TL", true},

		// In ccTLD before in ISO.
		{"GG", "GG", true},

		// Non-standard assignment of ccTLD to ISO code.
		{"GB", "UK", true},

		// Exceptionally reserved in ISO and valid ccTLD.
		{"UK", "UK", true},
		{"AC", "AC", true},
		{"EU", "EU", true},
		{"SU", "SU", true},

		// Exceptionally reserved in ISO and invalid ccTLD.
		{"CP", "ZZ", false},
		{"DG", "ZZ", false},
		{"EA", "ZZ", false},
		{"FX", "ZZ", false},
		{"IC", "ZZ", false},
		{"TA", "ZZ", false},

		// Transitionally reserved in ISO (e.g. deprecated) but valid ccTLD as
		// it is still being phased out.
		{"AN", "AN", true},
		{"TP", "TP", true},

		// Transitionally reserved in ISO (e.g. deprecated) and invalid ccTLD.
		// Defined in package language as it has a mapping in CLDR.
		{"BU", "ZZ", false},
		{"CS", "ZZ", false},
		{"NT", "ZZ", false},
		{"YU", "ZZ", false},
		{"ZR", "ZZ", false},
		// Not defined in package: SF.

		// Indeterminately reserved in ISO.
		// Defined in package language as it has a legacy mapping in CLDR.
		{"DY", "ZZ", false},
		{"RH", "ZZ", false},
		{"VD", "ZZ", false},
		// Not defined in package: EW, FL, JA, LF, PI, RA, RB, RC, RI, RL, RM,
		// RN, RP, WG, WL, WV, and YV.

		// Not assigned in ISO, but legacy definitions in CLDR.
		{"DD", "ZZ", false},
		{"YD", "ZZ", false},

		// Normal mappings but somewhat special status in ccTLD.
		{"BL", "BL", true},
		{"MF", "MF", true},
		{"BV", "BV", true},
		{"SJ", "SJ", true},

		// Have values when normalized, but not as is.
		{"QU", "ZZ", false},

		// ISO Private Use.
		{"AA", "ZZ", false},
		{"QM", "ZZ", false},
		{"QO", "ZZ", false},
		{"XA", "ZZ", false},
		{"XK", "ZZ", false}, // Sometimes used for Kosovo, but invalid ccTLD.
	} {
		if tt.in == "" {
			continue
		}

		r := MustParseRegion(tt.in)
		var want Region
		if tt.out != "ZZ" {
			want = MustParseRegion(tt.out)
		}
		tld, err := r.TLD()
		if got := err == nil; got != tt.ok {
			t.Errorf("error(%v): got %v; want %v", r, got, tt.ok)
		}
		if tld != want {
			t.Errorf("TLD(%v): got %v; want %v", r, tld, want)
		}
	}
}

func TestTypeForKey(t *testing.T) {
	tests := []struct{ key, in, out string }{
		{"co", "en", ""},
		{"co", "en-u-abc", ""},
		{"co", "en-u-co-phonebk", "phonebk"},
		{"co", "en-u-co-phonebk-cu-aud", "phonebk"},
		{"co", "x-foo-u-co-phonebk", ""},
		{"nu", "en-u-co-phonebk-nu-arabic", "arabic"},
		{"kc", "cmn-u-co-stroke", ""},
	}
	for _, tt := range tests {
		if v := Make(tt.in).TypeForKey(tt.key); v != tt.out {
			t.Errorf("%q[%q]: was %q; want %q", tt.in, tt.key, v, tt.out)
		}
	}
}

func TestSetTypeForKey(t *testing.T) {
	tests := []struct {
		key, value, in, out string
		err                 bool
	}{
		// replace existing value
		{"co", "pinyin", "en-u-co-phonebk", "en-u-co-pinyin", false},
		{"co", "pinyin", "en-u-co-phonebk-cu-xau", "en-u-co-pinyin-cu-xau", false},
		{"co", "pinyin", "en-u-co-phonebk-v-xx", "en-u-co-pinyin-v-xx", false},
		{"co", "pinyin", "en-u-co-phonebk-x-x", "en-u-co-pinyin-x-x", false},
		{"co", "pinyin", "en-u-co-x-x", "en-u-co-pinyin-x-x", false},
		{"nu", "arabic", "en-u-co-phonebk-nu-vaai", "en-u-co-phonebk-nu-arabic", false},
		{"nu", "arabic", "en-u-co-phonebk-nu", "en-u-co-phonebk-nu-arabic", false},
		// add to existing -u extension
		{"co", "pinyin", "en-u-ca-gregory", "en-u-ca-gregory-co-pinyin", false},
		{"co", "pinyin", "en-u-ca-gregory-nu-vaai", "en-u-ca-gregory-co-pinyin-nu-vaai", false},
		{"co", "pinyin", "en-u-ca-gregory-v-va", "en-u-ca-gregory-co-pinyin-v-va", false},
		{"co", "pinyin", "en-u-ca-gregory-x-a", "en-u-ca-gregory-co-pinyin-x-a", false},
		{"ca", "gregory", "en-u-co-pinyin", "en-u-ca-gregory-co-pinyin", false},
		// remove pair
		{"co", "", "en-u-co-phonebk", "en", false},
		{"co", "", "en-u-co", "en", false},
		{"co", "", "en-u-co-v", "en", false},
		{"co", "", "en-u-co-v-", "en", false},
		{"co", "", "en-u-ca-gregory-co-phonebk", "en-u-ca-gregory", false},
		{"co", "", "en-u-co-phonebk-nu-arabic", "en-u-nu-arabic", false},
		{"co", "", "en-u-co-nu-arabic", "en-u-nu-arabic", false},
		{"co", "", "en", "en", false},
		// add -u extension
		{"co", "pinyin", "en", "en-u-co-pinyin", false},
		{"co", "pinyin", "und", "und-u-co-pinyin", false},
		{"co", "pinyin", "en-a-aaa", "en-a-aaa-u-co-pinyin", false},
		{"co", "pinyin", "en-x-aaa", "en-u-co-pinyin-x-aaa", false},
		{"co", "pinyin", "en-v-aa", "en-u-co-pinyin-v-aa", false},
		{"co", "pinyin", "en-a-aaa-x-x", "en-a-aaa-u-co-pinyin-x-x", false},
		{"co", "pinyin", "en-a-aaa-v-va", "en-a-aaa-u-co-pinyin-v-va", false},
		// error on invalid values
		{"co", "pinyinxxx", "en", "en", true},
		{"co", "piny.n", "en", "en", true},
		{"co", "pinyinxxx", "en-a-aaa", "en-a-aaa", true},
		{"co", "pinyinxxx", "en-u-aaa", "en-u-aaa", true},
		{"co", "pinyinxxx", "en-u-aaa-co-pinyin", "en-u-aaa-co-pinyin", true},
		{"co", "pinyi.", "en-u-aaa-co-pinyin", "en-u-aaa-co-pinyin", true},
		{"col", "pinyin", "en", "en", true},
		{"co", "cu", "en", "en", true},
		// error when setting on a private use tag
		{"co", "phonebook", "x-foo", "x-foo", true},
	}
	for i, tt := range tests {
		tag := Make(tt.in)
		if v, err := tag.SetTypeForKey(tt.key, tt.value); v.String() != tt.out {
			t.Errorf("%d:%q[%q]=%q: was %q; want %q", i, tt.in, tt.key, tt.value, v, tt.out)
		} else if (err != nil) != tt.err {
			t.Errorf("%d:%q[%q]=%q: error was %v; want %v", i, tt.in, tt.key, tt.value, err != nil, tt.err)
		} else if val := v.TypeForKey(tt.key); err == nil && val != tt.value {
			t.Errorf("%d:%q[%q]==%q: was %v; want %v", i, tt.out, tt.key, tt.value, val, tt.value)
		}
		if len(tag.String()) <= 3 {
			// Simulate a tag for which the string has not been set.
			tag.str, tag.pExt, tag.pVariant = "", 0, 0
			if tag, err := tag.SetTypeForKey(tt.key, tt.value); err == nil {
				if val := tag.TypeForKey(tt.key); err == nil && val != tt.value {
					t.Errorf("%d:%q[%q]==%q: was %v; want %v", i, tt.out, tt.key, tt.value, val, tt.value)
				}
			}
		}
	}
}

func TestFindKeyAndType(t *testing.T) {
	// out is either the matched type in case of a match or the original
	// string up till the insertion point.
	tests := []struct {
		key     string
		hasExt  bool
		in, out string
	}{
		// Don't search past a private use extension.
		{"co", false, "en-x-foo-u-co-pinyin", "en"},
		{"co", false, "x-foo-u-co-pinyin", ""},
		{"co", false, "en-s-fff-x-foo", "en-s-fff"},
		// Insertion points in absence of -u extension.
		{"cu", false, "en", ""}, // t.str is ""
		{"cu", false, "en-v-va", "en"},
		{"cu", false, "en-a-va", "en-a-va"},
		{"cu", false, "en-a-va-v-va", "en-a-va"},
		{"cu", false, "en-x-a", "en"},
		// Tags with the -u extension.
		{"nu", true, "en-u-cu-nu", "en-u-cu"},
		{"cu", true, "en-u-cu-nu", "en-u"},
		{"co", true, "en-u-co-standard", "standard"},
		{"co", true, "yue-u-co-pinyin", "pinyin"},
		{"co", true, "en-u-co-abc", "abc"},
		{"co", true, "en-u-co-abc-def", "abc-def"},
		{"co", true, "en-u-co-abc-def-x-foo", "abc-def"},
		{"co", true, "en-u-co-standard-nu-arab", "standard"},
		{"co", true, "yue-u-co-pinyin-nu-arab", "pinyin"},
		// Insertion points.
		{"cu", true, "en-u-co-standard", "en-u-co-standard"},
		{"cu", true, "yue-u-co-pinyin-x-foo", "yue-u-co-pinyin"},
		{"cu", true, "en-u-co-abc", "en-u-co-abc"},
		{"cu", true, "en-u-nu-arabic", "en-u"},
		{"cu", true, "en-u-co-abc-def-nu-arabic", "en-u-co-abc-def"},
	}
	for i, tt := range tests {
		start, sep, end, hasExt := Make(tt.in).findTypeForKey(tt.key)
		if sep != end {
			res := tt.in[sep:end]
			if res != tt.out {
				t.Errorf("%d:%s: was %q; want %q", i, tt.in, res, tt.out)
			}
		} else {
			if hasExt != tt.hasExt {
				t.Errorf("%d:%s: hasExt was %v; want %v", i, tt.in, hasExt, tt.hasExt)
				continue
			}
			if tt.in[:start] != tt.out {
				t.Errorf("%d:%s: insertion point was %q; want %q", i, tt.in, tt.in[:start], tt.out)
			}
		}
	}
}

func TestParent(t *testing.T) {
	tests := []struct{ in, out string }{
		// Strip variants and extensions first
		{"de-u-co-phonebk", "de"},
		{"de-1994", "de"},
		{"de-Latn-1994", "de"}, // remove superfluous script.

		// Ensure the canonical Tag for an entry is in the chain for base-script
		// pairs.
		{"zh-Hans", "zh"},

		// Skip the script if it is the maximized version. CLDR files for the
		// skipped tag are always empty.
		{"zh-Hans-TW", "zh"},
		{"zh-Hans-CN", "zh"},

		// Insert the script if the maximized script is not the same as the
		// maximized script of the base language.
		{"zh-TW", "zh-Hant"},
		{"zh-HK", "zh-Hant"},
		{"zh-Hant-TW", "zh-Hant"},
		{"zh-Hant-HK", "zh-Hant"},

		// Non-default script skips to und.
		// CLDR
		{"az-Cyrl", "und"},
		{"bs-Cyrl", "und"},
		{"en-Dsrt", "und"},
		{"ha-Arab", "und"},
		{"mn-Mong", "und"},
		{"pa-Arab", "und"},
		{"shi-Latn", "und"},
		{"sr-Latn", "und"},
		{"uz-Arab", "und"},
		{"uz-Cyrl", "und"},
		{"vai-Latn", "und"},
		{"zh-Hant", "und"},
		// extra
		{"nl-Cyrl", "und"},

		// World english inherits from en-001.
		{"en-150", "en-001"},
		{"en-AU", "en-001"},
		{"en-BE", "en-001"},
		{"en-GG", "en-001"},
		{"en-GI", "en-001"},
		{"en-HK", "en-001"},
		{"en-IE", "en-001"},
		{"en-IM", "en-001"},
		{"en-IN", "en-001"},
		{"en-JE", "en-001"},
		{"en-MT", "en-001"},
		{"en-NZ", "en-001"},
		{"en-PK", "en-001"},
		{"en-SG", "en-001"},

		// Spanish in Latin-American countries have es-419 as parent.
		{"es-AR", "es-419"},
		{"es-BO", "es-419"},
		{"es-CL", "es-419"},
		{"es-CO", "es-419"},
		{"es-CR", "es-419"},
		{"es-CU", "es-419"},
		{"es-DO", "es-419"},
		{"es-EC", "es-419"},
		{"es-GT", "es-419"},
		{"es-HN", "es-419"},
		{"es-MX", "es-419"},
		{"es-NI", "es-419"},
		{"es-PA", "es-419"},
		{"es-PE", "es-419"},
		{"es-PR", "es-419"},
		{"es-PY", "es-419"},
		{"es-SV", "es-419"},
		{"es-US", "es-419"},
		{"es-UY", "es-419"},
		{"es-VE", "es-419"},
		// exceptions (according to CLDR)
		{"es-CW", "es"},

		// Inherit from pt-PT, instead of pt for these countries.
		{"pt-AO", "pt-PT"},
		{"pt-CV", "pt-PT"},
		{"pt-GW", "pt-PT"},
		{"pt-MO", "pt-PT"},
		{"pt-MZ", "pt-PT"},
		{"pt-ST", "pt-PT"},
		{"pt-TL", "pt-PT"},
	}
	for _, tt := range tests {
		tag := MustParse(tt.in)
		if p := MustParse(tt.out); p != tag.Parent() {
			t.Errorf("%s: was %v; want %v", tt.in, tag.Parent(), p)
		}
	}
}

var (
	// Tags without error that don't need to be changed.
	benchBasic = []string{
		"en",
		"en-Latn",
		"en-GB",
		"za",
		"zh-Hant",
		"zh",
		"zh-HK",
		"ar-MK",
		"en-CA",
		"fr-CA",
		"fr-CH",
		"fr",
		"lv",
		"he-IT",
		"tlh",
		"ja",
		"ja-Jpan",
		"ja-Jpan-JP",
		"de-1996",
		"de-CH",
		"sr",
		"sr-Latn",
	}
	// Tags with extensions, not changes required.
	benchExt = []string{
		"x-a-b-c-d",
		"x-aa-bbbb-cccccccc-d",
		"en-x_cc-b-bbb-a-aaa",
		"en-c_cc-b-bbb-a-aaa-x-x",
		"en-u-co-phonebk",
		"en-Cyrl-u-co-phonebk",
		"en-US-u-co-phonebk-cu-xau",
		"en-nedix-u-co-phonebk",
		"en-t-t0-abcd",
		"en-t-nl-latn",
		"en-t-t0-abcd-x-a",
	}
	// Change, but not memory allocation required.
	benchSimpleChange = []string{
		"EN",
		"i-klingon",
		"en-latn",
		"zh-cmn-Hans-CN",
		"iw-NL",
	}
	// Change and memory allocation required.
	benchChangeAlloc = []string{
		"en-c_cc-b-bbb-a-aaa",
		"en-u-cu-xua-co-phonebk",
		"en-u-cu-xua-co-phonebk-a-cd",
		"en-u-def-abc-cu-xua-co-phonebk",
		"en-t-en-Cyrl-NL-1994",
		"en-t-en-Cyrl-NL-1994-t0-abc-def",
	}
	// Tags that result in errors.
	benchErr = []string{
		// IllFormed
		"x_A.-B-C_D",
		"en-u-cu-co-phonebk",
		"en-u-cu-xau-co",
		"en-t-nl-abcd",
		// Invalid
		"xx",
		"nl-Uuuu",
		"nl-QB",
	}
	benchChange = append(benchSimpleChange, benchChangeAlloc...)
	benchAll    = append(append(append(benchBasic, benchExt...), benchChange...), benchErr...)
)

func doParse(b *testing.B, tag []string) {
	for i := 0; i < b.N; i++ {
		// Use the modulo instead of looping over all tags so that we get a somewhat
		// meaningful ns/op.
		Parse(tag[i%len(tag)])
	}
}

func BenchmarkParse(b *testing.B) {
	doParse(b, benchAll)
}

func BenchmarkParseBasic(b *testing.B) {
	doParse(b, benchBasic)
}

func BenchmarkParseError(b *testing.B) {
	doParse(b, benchErr)
}

func BenchmarkParseSimpleChange(b *testing.B) {
	doParse(b, benchSimpleChange)
}

func BenchmarkParseChangeAlloc(b *testing.B) {
	doParse(b, benchChangeAlloc)
}
