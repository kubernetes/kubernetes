// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"testing"

	"golang.org/x/text/language"
)

func TestParents(t *testing.T) {
	testCases := []struct {
		tag, parent string
	}{
		{"af", "und"},
		{"en", "und"},
		{"en-001", "en"},
		{"en-AU", "en-001"},
		{"en-US", "en"},
		{"en-US-u-va-posix", "en-US"},
		{"ca-ES-valencia", "ca-ES"},
	}
	for _, tc := range testCases {
		tag, ok := language.CompactIndex(language.MustParse(tc.tag))
		if !ok {
			t.Fatalf("Could not get index of flag %s", tc.tag)
		}
		want, ok := language.CompactIndex(language.MustParse(tc.parent))
		if !ok {
			t.Fatalf("Could not get index of parent %s of tag %s", tc.parent, tc.tag)
		}
		if got := int(Parent[tag]); got != want {
			t.Errorf("Parent[%s] = %d; want %d (%s)", tc.tag, got, want, tc.parent)
		}
	}
}
