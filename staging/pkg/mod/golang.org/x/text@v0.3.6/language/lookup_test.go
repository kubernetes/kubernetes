// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"testing"
)

func TestRegionID(t *testing.T) {
	tests := []struct {
		in, out string
	}{
		{"_  ", ""},
		{"_000", ""},
		{"419", "419"},
		{"AA", "AA"},
		{"ATF", "TF"},
		{"HV", "HV"},
		{"CT", "CT"},
		{"DY", "DY"},
		{"IC", "IC"},
		{"FQ", "FQ"},
		{"JT", "JT"},
		{"ZZ", "ZZ"},
		{"EU", "EU"},
		{"QO", "QO"},
		{"FX", "FX"},
	}
	for i, tt := range tests {
		if tt.in[0] == '_' {
			id := tt.in[1:]
			if _, err := ParseRegion(id); err == nil {
				t.Errorf("%d:err(%s): found nil; want error", i, id)
			}
			continue
		}
		want, _ := ParseRegion(tt.in)
		if s := want.String(); s != tt.out {
			t.Errorf("%d:%s: found %q; want %q", i, tt.in, s, tt.out)
		}
		if len(tt.in) == 2 {
			want, _ := ParseRegion(tt.in)
			if s := want.String(); s != tt.out {
				t.Errorf("%d:getISO2(%s): found %q; want %q", i, tt.in, s, tt.out)
			}
		}
	}
}

func TestRegionISO3(t *testing.T) {
	tests := []struct {
		from, iso3, to string
	}{
		{"  ", "ZZZ", "ZZ"},
		{"000", "ZZZ", "ZZ"},
		{"AA", "AAA", ""},
		{"CT", "CTE", ""},
		{"DY", "DHY", ""},
		{"EU", "QUU", ""},
		{"HV", "HVO", ""},
		{"IC", "ZZZ", "ZZ"},
		{"JT", "JTN", ""},
		{"PZ", "PCZ", ""},
		{"QU", "QUU", "EU"},
		{"QO", "QOO", ""},
		{"YD", "YMD", ""},
		{"FQ", "ATF", "TF"},
		{"TF", "ATF", ""},
		{"FX", "FXX", ""},
		{"ZZ", "ZZZ", ""},
		{"419", "ZZZ", "ZZ"},
	}
	for _, tt := range tests {
		r, _ := ParseRegion(tt.from)
		if s := r.ISO3(); s != tt.iso3 {
			t.Errorf("iso3(%q): found %q; want %q", tt.from, s, tt.iso3)
		}
		if tt.iso3 == "" {
			continue
		}
		want := tt.to
		if tt.to == "" {
			want = tt.from
		}
		r, _ = ParseRegion(want)
		if id, _ := ParseRegion(tt.iso3); id != r {
			t.Errorf("%s: found %q; want %q", tt.iso3, id, want)
		}
	}
}

func TestRegionM49(t *testing.T) {
	fromTests := []struct {
		m49 int
		id  string
	}{
		{0, ""},
		{-1, ""},
		{1000, ""},
		{10000, ""},

		{001, "001"},
		{104, "MM"},
		{180, "CD"},
		{230, "ET"},
		{231, "ET"},
		{249, "FX"},
		{250, "FR"},
		{276, "DE"},
		{278, "DD"},
		{280, "DE"},
		{419, "419"},
		{626, "TL"},
		{736, "SD"},
		{840, "US"},
		{854, "BF"},
		{891, "CS"},
		{899, ""},
		{958, "AA"},
		{966, "QT"},
		{967, "EU"},
		{999, "ZZ"},
	}
	for _, tt := range fromTests {
		id, err := EncodeM49(tt.m49)
		if want, have := err != nil, tt.id == ""; want != have {
			t.Errorf("error(%d): have %v; want %v", tt.m49, have, want)
			continue
		}
		r, _ := ParseRegion(tt.id)
		if r != id {
			t.Errorf("region(%d): have %s; want %s", tt.m49, id, r)
		}
	}

	toTests := []struct {
		m49 int
		id  string
	}{
		{0, "000"},
		{0, "IC"}, // Some codes don't have an ID

		{001, "001"},
		{104, "MM"},
		{104, "BU"},
		{180, "CD"},
		{180, "ZR"},
		{231, "ET"},
		{250, "FR"},
		{249, "FX"},
		{276, "DE"},
		{278, "DD"},
		{419, "419"},
		{626, "TL"},
		{626, "TP"},
		{729, "SD"},
		{826, "GB"},
		{840, "US"},
		{854, "BF"},
		{891, "YU"},
		{891, "CS"},
		{958, "AA"},
		{966, "QT"},
		{967, "EU"},
		{967, "QU"},
		{999, "ZZ"},
		// For codes that don't have an M49 code use the replacement value,
		// if available.
		{854, "HV"}, // maps to Burkino Faso
	}
	for _, tt := range toTests {
		r, _ := ParseRegion(tt.id)
		if r.M49() != tt.m49 {
			t.Errorf("m49(%q): have %d; want %d", tt.id, r.M49(), tt.m49)
		}
	}
}

func TestRegionDeprecation(t *testing.T) {
	tests := []struct{ in, out string }{
		{"BU", "MM"},
		{"BUR", "MM"},
		{"CT", "KI"},
		{"DD", "DE"},
		{"DDR", "DE"},
		{"DY", "BJ"},
		{"FX", "FR"},
		{"HV", "BF"},
		{"JT", "UM"},
		{"MI", "UM"},
		{"NH", "VU"},
		{"NQ", "AQ"},
		{"PU", "UM"},
		{"PZ", "PA"},
		{"QU", "EU"},
		{"RH", "ZW"},
		{"TP", "TL"},
		{"UK", "GB"},
		{"VD", "VN"},
		{"WK", "UM"},
		{"YD", "YE"},
		{"NL", "NL"},
	}
	for _, tt := range tests {
		rIn, _ := ParseRegion(tt.in)
		rOut, _ := ParseRegion(tt.out)
		r := rIn.Canonicalize()
		if rOut == rIn && r.String() == "ZZ" {
			t.Errorf("%s: was %q; want %q", tt.in, r, tt.in)
		}
		if rOut != rIn && r != rOut {
			t.Errorf("%s: was %q; want %q", tt.in, r, tt.out)
		}

	}
}

func TestIsPrivateUse(t *testing.T) {
	type test struct {
		s       string
		private bool
	}
	tests := []test{
		{"en", false},
		{"und", false},
		{"pzn", false},
		{"qaa", true},
		{"qtz", true},
		{"qua", false},
	}
	for i, tt := range tests {
		x, _ := ParseBase(tt.s)
		if b := x.IsPrivateUse(); b != tt.private {
			t.Errorf("%d: langID.IsPrivateUse(%s) was %v; want %v", i, tt.s, b, tt.private)
		}
	}
	tests = []test{
		{"001", false},
		{"419", false},
		{"899", false},
		{"900", false},
		{"957", false},
		{"958", true},
		{"AA", true},
		{"AC", false},
		{"EU", false}, // CLDR grouping, exceptionally reserved in ISO.
		{"QU", true},  // Canonicalizes to EU, User-assigned in ISO.
		{"QO", true},  // CLDR grouping, User-assigned in ISO.
		{"QA", false},
		{"QM", true},
		{"QZ", true},
		{"XA", true},
		{"XK", true}, // Assigned to Kosovo in CLDR, User-assigned in ISO.
		{"XZ", true},
		{"ZW", false},
		{"ZZ", true},
	}
	for i, tt := range tests {
		x, _ := ParseRegion(tt.s)
		if b := x.IsPrivateUse(); b != tt.private {
			t.Errorf("%d: regionID.IsPrivateUse(%s) was %v; want %v", i, tt.s, b, tt.private)
		}
	}
	tests = []test{
		{"Latn", false},
		{"Laaa", false}, // invalid
		{"Qaaa", true},
		{"Qabx", true},
		{"Qaby", false},
		{"Zyyy", false},
		{"Zzzz", false},
	}
	for i, tt := range tests {
		x, _ := ParseScript(tt.s)
		if b := x.IsPrivateUse(); b != tt.private {
			t.Errorf("%d: scriptID.IsPrivateUse(%s) was %v; want %v", i, tt.s, b, tt.private)
		}
	}
}
