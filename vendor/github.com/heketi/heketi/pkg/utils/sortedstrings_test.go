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
	"sort"
	"testing"
)

func TestSortedStringsHas(t *testing.T) {
	s := sort.StringSlice{"z", "b", "a"}
	s.Sort()
	tests.Assert(t, len(s) == 3)
	tests.Assert(t, s[0] == "a")
	tests.Assert(t, s[1] == "b")
	tests.Assert(t, s[2] == "z")

	tests.Assert(t, SortedStringHas(s, "a"))
	tests.Assert(t, SortedStringHas(s, "b"))
	tests.Assert(t, SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))
}

func TestSortedStringsDelete(t *testing.T) {
	s := sort.StringSlice{"z", "b", "a"}
	s.Sort()
	tests.Assert(t, len(s) == 3)
	tests.Assert(t, s[0] == "a")
	tests.Assert(t, s[1] == "b")
	tests.Assert(t, s[2] == "z")

	tests.Assert(t, SortedStringHas(s, "a"))
	tests.Assert(t, SortedStringHas(s, "b"))
	tests.Assert(t, SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))

	s = SortedStringsDelete(s, "notthere")
	tests.Assert(t, len(s) == 3)
	s = SortedStringsDelete(s, "zzzznotthere")
	tests.Assert(t, len(s) == 3)
	s = SortedStringsDelete(s, "1azzzznotthere")
	tests.Assert(t, len(s) == 3)
	tests.Assert(t, SortedStringHas(s, "a"))
	tests.Assert(t, SortedStringHas(s, "b"))
	tests.Assert(t, SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))

	s = SortedStringsDelete(s, "z")
	tests.Assert(t, len(s) == 2)
	tests.Assert(t, SortedStringHas(s, "a"))
	tests.Assert(t, SortedStringHas(s, "b"))
	tests.Assert(t, !SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))

	s = SortedStringsDelete(s, "a")
	tests.Assert(t, len(s) == 1)
	tests.Assert(t, !SortedStringHas(s, "a"))
	tests.Assert(t, SortedStringHas(s, "b"))
	tests.Assert(t, !SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))

	s = SortedStringsDelete(s, "b")
	tests.Assert(t, len(s) == 0)
	tests.Assert(t, !SortedStringHas(s, "a"))
	tests.Assert(t, !SortedStringHas(s, "b"))
	tests.Assert(t, !SortedStringHas(s, "z"))
	tests.Assert(t, !SortedStringHas(s, "c"))
	tests.Assert(t, !SortedStringHas(s, "zz"))

}
