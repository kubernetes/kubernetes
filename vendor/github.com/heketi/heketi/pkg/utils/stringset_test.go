//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package utils

import (
	"github.com/heketi/tests"
	"testing"
)

func TestNewStringSet(t *testing.T) {
	s := NewStringSet()
	tests.Assert(t, s.Set != nil)
	tests.Assert(t, len(s.Set) == 0)
}

func TestStringSet(t *testing.T) {
	s := NewStringSet()

	s.Add("one")
	s.Add("two")
	s.Add("three")
	tests.Assert(t, s.Len() == 3)
	tests.Assert(t, SortedStringHas(s.Set, "one"))
	tests.Assert(t, SortedStringHas(s.Set, "two"))
	tests.Assert(t, SortedStringHas(s.Set, "three"))

	s.Add("one")
	tests.Assert(t, s.Len() == 3)
	tests.Assert(t, SortedStringHas(s.Set, "one"))
	tests.Assert(t, SortedStringHas(s.Set, "two"))
	tests.Assert(t, SortedStringHas(s.Set, "three"))

	s.Add("three")
	tests.Assert(t, s.Len() == 3)
	tests.Assert(t, SortedStringHas(s.Set, "one"))
	tests.Assert(t, SortedStringHas(s.Set, "two"))
	tests.Assert(t, SortedStringHas(s.Set, "three"))

	s.Add("four")
	tests.Assert(t, s.Len() == 4)
	tests.Assert(t, SortedStringHas(s.Set, "one"))
	tests.Assert(t, SortedStringHas(s.Set, "two"))
	tests.Assert(t, SortedStringHas(s.Set, "three"))
	tests.Assert(t, SortedStringHas(s.Set, "four"))
}
