// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

func TestUnique(t *testing.T) {
	testCases := []struct {
		in, want string
	}{
		{"", "[]"},
		{"en", "[en]"},
		{"en en", "[en]"},
		{"en en en", "[en]"},
		{"en-u-cu-eur en", "[en en-u-cu-eur]"},
		{"nl en", "[en nl]"},
		{"pt-Pt pt", "[pt pt-PT]"},
	}
	for _, tc := range testCases {
		tags := []language.Tag{}
		for _, s := range strings.Split(tc.in, " ") {
			if s != "" {
				tags = append(tags, language.MustParse(s))
			}
		}
		if got := fmt.Sprint(UniqueTags(tags)); got != tc.want {
			t.Errorf("Unique(%s) = %s; want %s", tc.in, got, tc.want)
		}
	}
}
