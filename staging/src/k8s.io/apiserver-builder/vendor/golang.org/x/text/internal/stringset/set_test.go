// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stringset

import "testing"

func TestStringSet(t *testing.T) {
	testCases := [][]string{
		{""},
		{"∫"},
		{"a", "b", "c"},
		{"", "a", "bb", "ccc"},
		{"    ", "aaa", "bb", "c"},
	}
	test := func(tc int, b *Builder) {
		set := b.Set()
		if set.Len() != len(testCases[tc]) {
			t.Errorf("%d:Len() = %d; want %d", tc, set.Len(), len(testCases[tc]))
		}
		for i, s := range testCases[tc] {
			if x := b.Index(s); x != i {
				t.Errorf("%d:Index(%q) = %d; want %d", tc, s, x, i)
			}
			if p := Search(&set, s); p != i {
				t.Errorf("%d:Search(%q) = %d; want %d", tc, s, p, i)
			}
			if set.Elem(i) != s {
				t.Errorf("%d:Elem(%d) = %s; want %s", tc, i, set.Elem(i), s)
			}
		}
		if p := Search(&set, "apple"); p != -1 {
			t.Errorf(`%d:Search("apple") = %d; want -1`, tc, p)
		}
	}
	for i, tc := range testCases {
		b := NewBuilder()
		for _, s := range tc {
			b.Add(s)
		}
		b.Add(tc...)
		test(i, b)
	}
	for i, tc := range testCases {
		b := NewBuilder()
		b.Add(tc...)
		for _, s := range tc {
			b.Add(s)
		}
		test(i, b)
	}
}
