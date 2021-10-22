// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"fmt"
	"reflect"
	"testing"

	"golang.org/x/text/internal/language"
)

func TestSupported(t *testing.T) {
	// To prove the results are correct for a type, we test that the number of
	// results is identical to the number of results on record, that all results
	// are distinct and that all results are valid.
	tests := map[string]int{
		"BaseLanguages": language.NumLanguages,
		"Scripts":       language.NumScripts,
		"Regions":       language.NumRegions,
		"Tags":          0,
	}
	sup := reflect.ValueOf(Supported)
	for name, num := range tests {
		v := sup.MethodByName(name).Call(nil)[0]
		if n := v.Len(); n != num {
			t.Errorf("len(%s()) was %d; want %d", name, n, num)
		}
		dup := make(map[string]bool)
		for i := 0; i < v.Len(); i++ {
			x := v.Index(i).Interface()
			// An invalid value will either cause a crash or result in a
			// duplicate when passed to Sprint.
			s := fmt.Sprint(x)
			if dup[s] {
				t.Errorf("%s: duplicate entry %q", name, s)
			}
			dup[s] = true
		}
		if len(dup) != v.Len() {
			t.Errorf("%s: # unique entries was %d; want %d", name, len(dup), v.Len())
		}
	}
}

func TestNewCoverage(t *testing.T) {
	bases := []Base{Base{0}, Base{3}, Base{7}}
	scripts := []Script{Script{11}, Script{17}, Script{23}}
	regions := []Region{Region{101}, Region{103}, Region{107}}
	tags := []Tag{Make("pt"), Make("en"), Make("en-GB"), Make("en-US"), Make("pt-PT")}
	fbases := func() []Base { return bases }
	fscripts := func() []Script { return scripts }
	fregions := func() []Region { return regions }
	ftags := func() []Tag { return tags }

	tests := []struct {
		desc    string
		list    []interface{}
		bases   []Base
		scripts []Script
		regions []Region
		tags    []Tag
	}{
		{
			desc: "empty",
		},
		{
			desc:  "bases",
			list:  []interface{}{bases},
			bases: bases,
		},
		{
			desc:    "scripts",
			list:    []interface{}{scripts},
			scripts: scripts,
		},
		{
			desc:    "regions",
			list:    []interface{}{regions},
			regions: regions,
		},
		{
			desc:  "bases derives from tags",
			list:  []interface{}{tags},
			bases: []Base{Base{_en}, Base{_pt}},
			tags:  tags,
		},
		{
			desc:  "tags and bases",
			list:  []interface{}{tags, bases},
			bases: bases,
			tags:  tags,
		},
		{
			desc:    "fully specified",
			list:    []interface{}{tags, bases, scripts, regions},
			bases:   bases,
			scripts: scripts,
			regions: regions,
			tags:    tags,
		},
		{
			desc:  "bases func",
			list:  []interface{}{fbases},
			bases: bases,
		},
		{
			desc:    "scripts func",
			list:    []interface{}{fscripts},
			scripts: scripts,
		},
		{
			desc:    "regions func",
			list:    []interface{}{fregions},
			regions: regions,
		},
		{
			desc:  "tags func",
			list:  []interface{}{ftags},
			bases: []Base{Base{_en}, Base{_pt}},
			tags:  tags,
		},
		{
			desc:  "tags and bases",
			list:  []interface{}{ftags, fbases},
			bases: bases,
			tags:  tags,
		},
		{
			desc:    "fully specified",
			list:    []interface{}{ftags, fbases, fscripts, fregions},
			bases:   bases,
			scripts: scripts,
			regions: regions,
			tags:    tags,
		},
	}

	for i, tt := range tests {
		l := NewCoverage(tt.list...)
		if a := l.BaseLanguages(); !reflect.DeepEqual(a, tt.bases) {
			t.Errorf("%d:%s: BaseLanguages was %v; want %v", i, tt.desc, a, tt.bases)
		}
		if a := l.Scripts(); !reflect.DeepEqual(a, tt.scripts) {
			t.Errorf("%d:%s: Scripts was %v; want %v", i, tt.desc, a, tt.scripts)
		}
		if a := l.Regions(); !reflect.DeepEqual(a, tt.regions) {
			t.Errorf("%d:%s: Regions was %v; want %v", i, tt.desc, a, tt.regions)
		}
		if a := l.Tags(); !reflect.DeepEqual(a, tt.tags) {
			t.Errorf("%d:%s: Tags was %v; want %v", i, tt.desc, a, tt.tags)
		}
	}
}
