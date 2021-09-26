// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldr

import "testing"

func TestParseDraft(t *testing.T) {
	tests := []struct {
		in    string
		draft Draft
		err   bool
	}{
		{"unconfirmed", Unconfirmed, false},
		{"provisional", Provisional, false},
		{"contributed", Contributed, false},
		{"approved", Approved, false},
		{"", Approved, false},
		{"foo", Approved, true},
	}
	for _, tt := range tests {
		if d, err := ParseDraft(tt.in); d != tt.draft || (err != nil) != tt.err {
			t.Errorf("%q: was %v, %v; want %v, %v", tt.in, d, err != nil, tt.draft, tt.err)
		}
	}
}
