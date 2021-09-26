// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import "testing"

// TODO: most of Parser is tested in x/message. Move some tests here.

func TestParsenum(t *testing.T) {
	testCases := []struct {
		s          string
		start, end int
		num        int
		isnum      bool
		newi       int
	}{
		{"a123", 0, 4, 0, false, 0},
		{"1234", 1, 1, 0, false, 1},
		{"123a", 0, 4, 123, true, 3},
		{"12a3", 0, 4, 12, true, 2},
		{"1234", 0, 4, 1234, true, 4},
		{"1a234", 1, 3, 0, false, 1},
	}
	for _, tt := range testCases {
		num, isnum, newi := parsenum(tt.s, tt.start, tt.end)
		if num != tt.num || isnum != tt.isnum || newi != tt.newi {
			t.Errorf("parsenum(%q, %d, %d) = %d, %v, %d, want %d, %v, %d", tt.s, tt.start, tt.end, num, isnum, newi, tt.num, tt.isnum, tt.newi)
		}
	}
}
