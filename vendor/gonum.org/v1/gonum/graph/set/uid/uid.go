// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package uid implements unique ID provision for graphs.
package uid

import (
	"math"

	"gonum.org/v1/gonum/graph/internal/set"
)

// Max is the maximum ID value.
const Max = math.MaxInt64

// Set implements available ID storage.
type Set struct {
	maxID      int64
	used, free set.Int64s
}

// NewSet returns a new Set.
func NewSet() *Set {
	return &Set{maxID: -1, used: make(set.Int64s), free: make(set.Int64s)}
}

// NewID returns a new unique ID. The ID returned is not considered used
// until passed in a call to use.
func (s *Set) NewID() int64 {
	for id := range s.free {
		return id
	}
	if s.maxID != math.MaxInt64 {
		return s.maxID + 1
	}
	for id := int64(0); id <= s.maxID; id++ {
		if !s.used.Has(id) {
			return id
		}
	}
	panic("unreachable")
}

// Use adds the id to the used IDs in the Set.
func (s *Set) Use(id int64) {
	s.used.Add(id)
	s.free.Remove(id)
	if id > s.maxID {
		s.maxID = id
	}
}

// Release frees the id for reuse.
func (s *Set) Release(id int64) {
	s.free.Add(id)
	s.used.Remove(id)
}
