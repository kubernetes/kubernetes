// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compact

import (
	"reflect"
	"testing"

	"golang.org/x/text/internal/language"
)

func mustParse(s string) Tag {
	t, err := language.Parse(s)
	if err != nil {
		panic(err)
	}
	return Make(t)
}

func TestTagSize(t *testing.T) {
	id := Tag{}
	typ := reflect.TypeOf(id)
	if typ.Size() > 24 {
		t.Errorf("size of Tag was %d; want 24", typ.Size())
	}
}

func TestNoPublic(t *testing.T) {
	noExportedField(t, reflect.TypeOf(Tag{}))
}

func noExportedField(t *testing.T, typ reflect.Type) {
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.PkgPath == "" {
			t.Errorf("Tag may not have exported fields, but has field %q", f.Name)
		}
		if f.Anonymous {
			noExportedField(t, f.Type)
		}
	}
}

func TestEquality(t *testing.T) {
	for i, tt := range parseTests() {
		s := tt.in
		tag := mk(s)
		t1 := mustParse(tag.Tag().String())
		if tag != t1 {
			t.Errorf("%d:%s: equality test 1 failed\n got: %#v\nwant: %#v)", i, s, t1, tag)
		}
	}
}

type compactTest struct {
	tag   string
	index ID
	ok    bool
}

var compactTests = []compactTest{
	// TODO: these values will change with each CLDR update. This issue
	// will be solved if we decide to fix the indexes.
	{"und", undIndex, true},
	{"ca-ES-valencia", caESvalenciaIndex, true},
	{"ca-ES-valencia-u-va-posix", caESvalenciaIndex, false},
	{"ca-ES-valencia-u-co-phonebk", caESvalenciaIndex, false},
	{"ca-ES-valencia-u-co-phonebk-va-posix", caESvalenciaIndex, false},
	{"x-klingon", 0, false},
	{"en-US", enUSIndex, true},
	{"en-US-u-va-posix", enUSuvaposixIndex, true},
	{"en", enIndex, true},
	{"en-u-co-phonebk", enIndex, false},
	{"en-001", en001Index, true},
	{"zh-Hant-HK", zhHantHKIndex, true},
	{"zh-HK", zhHantHKIndex, false}, // maximized to zh-Hant-HK
	{"nl-Beng", 0, false},           // parent skips script
	{"nl-NO", nlIndex, false},       // region is ignored
	{"nl-Latn-NO", nlIndex, false},
	{"nl-Latn-NO-u-co-phonebk", nlIndex, false},
	{"nl-Latn-NO-valencia", nlIndex, false},
	{"nl-Latn-NO-oxendict", nlIndex, false},
	{"sh", shIndex, true}, // From plural rules.
}

func TestLanguageID(t *testing.T) {
	tests := append(compactTests, []compactTest{
		{"en-GB", enGBIndex, true},
		{"en-GB-u-rg-uszzzz", enGBIndex, true},
		{"en-GB-u-rg-USZZZZ", enGBIndex, true},
		{"en-GB-u-rg-uszzzz-va-posix", enGBIndex, false},
		{"en-GB-u-co-phonebk-rg-uszzzz", enGBIndex, false},
		// Invalid region specifications are ignored.
		{"en-GB-u-rg-usz-va-posix", enGBIndex, false},
		{"en-GB-u-co-phonebk-rg-usz", enGBIndex, false},
	}...)
	for _, tt := range tests {
		x, ok := LanguageID(mustParse(tt.tag))
		if ID(x) != tt.index || ok != tt.ok {
			t.Errorf("%s: got %d, %v; want %d %v", tt.tag, x, ok, tt.index, tt.ok)
		}
	}
}

func TestRegionalID(t *testing.T) {
	tests := append(compactTests, []compactTest{
		{"en-GB", enGBIndex, true},
		{"en-GB-u-rg-uszzzz", enUSIndex, true},
		{"en-GB-u-rg-USZZZZ", enUSIndex, true},
		// TODO: use different exact values for language and regional tag?
		{"en-GB-u-rg-uszzzz-va-posix", enUSuvaposixIndex, false},
		{"en-GB-u-co-phonebk-rg-uszzzz-va-posix", enUSuvaposixIndex, false},
		{"en-GB-u-co-phonebk-rg-uszzzz", enUSIndex, false},
		// Invalid region specifications are ignored.
		{"en-GB-u-rg-usz-va-posix", enGBIndex, false},
		{"en-GB-u-co-phonebk-rg-usz", enGBIndex, false},
	}...)
	for _, tt := range tests {
		x, ok := RegionalID(mustParse(tt.tag))
		if ID(x) != tt.index || ok != tt.ok {
			t.Errorf("%s: got %d, %v; want %d %v", tt.tag, x, ok, tt.index, tt.ok)
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

		{"en-GB-u-co-phonebk-rg-uszzzz", "en-GB"},
		{"en-GB-u-rg-uszzzz", "en-GB"},
		{"en-US-u-va-posix", "en-US"},

		// Difference between language and regional tag.
		{"ca-ES-valencia", "ca-ES"},
		{"ca-ES-valencia-u-rg-ptzzzz", "ca-ES"}, // t.full != nil
		{"en-US-u-va-variant", "en-US"},
		{"en-u-va-variant", "en"}, // t.full != nil
		{"en-u-rg-gbzzzz", "en"},
		{"en-US-u-rg-gbzzzz", "en-US"},
		{"nl-US-u-rg-gbzzzz", "nl-US"}, // t.full != nil
	}
	for _, tt := range tests {
		tag := mustParse(tt.in)
		if p := mustParse(tt.out); p != tag.Parent() {
			t.Errorf("%s: was %v; want %v", tt.in, tag.Parent(), p)
		}
	}
}
