// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"strings"
	"testing"

	"golang.org/x/text/language"
)

func TestInheritanceMatcher(t *testing.T) {
	for i, tt := range []struct {
		haveTags string
		wantTags string
		match    string
		conf     language.Confidence
	}{
		{"und,en,en-US", "en-US", "en-US", language.Exact}, // most specific match
		{"zh-Hant,zh", "zh-TW", "zh-Hant", language.High},  // zh-TW implies Hant.
		{"und,zh", "zh-TW", "und", language.High},          // zh-TW does not match zh.
		{"zh", "zh-TW", "und", language.No},                // zh-TW does not match zh.
		{"iw,en,nl", "he", "he", language.Exact},           // matches after canonicalization
		{"he,en,nl", "iw", "he", language.Exact},           // matches after canonicalization
		// Prefer first match over more specific match for various reasons:
		// a) consistency of user interface is more important than an exact match,
		// b) _if_ und is specified, it should be considered a correct and useful match,
		// Note that a call to this Match will almost always be with a single tag.
		{"und,en,en-US", "he,en-US", "und", language.High},
	} {
		have := parseTags(tt.haveTags)
		m := NewInheritanceMatcher(have)
		tag, index, conf := m.Match(parseTags(tt.wantTags)...)
		want := language.Raw.Make(tt.match)
		if tag != want {
			t.Errorf("%d:tag: got %q; want %q", i, tag, want)
		}
		if conf != language.No {
			if got, _ := language.All.Canonicalize(have[index]); got != want {
				t.Errorf("%d:index: got %q; want %q ", i, got, want)
			}
		}
		if conf != tt.conf {
			t.Errorf("%d:conf: got %v; want %v", i, conf, tt.conf)
		}
	}
}

func parseTags(list string) (out []language.Tag) {
	for _, s := range strings.Split(list, ",") {
		out = append(out, language.Raw.Make(strings.TrimSpace(s)))
	}
	return out
}
