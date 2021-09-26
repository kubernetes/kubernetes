// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"strings"
	"testing"
)

func parseBase(s string) Language {
	if s == "" {
		return 0
	}
	return MustParseBase(s)
}

func parseScript(s string) Script {
	if s == "" {
		return 0
	}
	return MustParseScript(s)
}

func parseRegion(s string) Region {
	if s == "" {
		return 0
	}
	return MustParseRegion(s)
}

func TestBuilder(t *testing.T) {
	partChecks(t, func(t *testing.T, tt *parseTest) (id Tag, skip bool) {
		tag := Make(tt.in)
		b := Builder{}
		b.SetTag(Tag{
			LangID:   parseBase(tt.lang),
			ScriptID: parseScript(tt.script),
			RegionID: parseRegion(tt.region),
		})
		if tt.variants != "" {
			b.AddVariant(strings.Split(tt.variants, "-")...)
		}
		for _, e := range tag.Extensions() {
			b.AddExt(e)
		}
		got := b.Make()
		if got != tag {
			t.Errorf("%s: got %v; want %v", tt.in, got, tag)
		}
		return got, false
	})
}

func TestSetTag(t *testing.T) {
	partChecks(t, func(t *testing.T, tt *parseTest) (id Tag, skip bool) {
		tag := Make(tt.in)
		b := Builder{}
		b.SetTag(tag)
		got := b.Make()
		if got != tag {
			t.Errorf("%s: got %v; want %v", tt.in, got, tag)
		}
		return got, false
	})
}
