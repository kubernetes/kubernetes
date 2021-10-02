// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.2
// +build !go1.2

package language

import "sort"

func sortStable(s sort.Interface) {
	ss := stableSort{
		s:   s,
		pos: make([]int, s.Len()),
	}
	for i := range ss.pos {
		ss.pos[i] = i
	}
	sort.Sort(&ss)
}

type stableSort struct {
	s   sort.Interface
	pos []int
}

func (s *stableSort) Len() int {
	return len(s.pos)
}

func (s *stableSort) Less(i, j int) bool {
	return s.s.Less(i, j) || !s.s.Less(j, i) && s.pos[i] < s.pos[j]
}

func (s *stableSort) Swap(i, j int) {
	s.s.Swap(i, j)
	s.pos[i], s.pos[j] = s.pos[j], s.pos[i]
}
