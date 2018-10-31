// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hash

// Roller provides an interface for rolling hashes. The hash value will become
// valid after hash has been called Len times.
type Roller interface {
	Len() int
	RollByte(x byte) uint64
}

// Hashes computes all hash values for the array p. Note that the state of the
// roller is changed.
func Hashes(r Roller, p []byte) []uint64 {
	n := r.Len()
	if len(p) < n {
		return nil
	}
	h := make([]uint64, len(p)-n+1)
	for i := 0; i < n-1; i++ {
		r.RollByte(p[i])
	}
	for i := range h {
		h[i] = r.RollByte(p[i+n-1])
	}
	return h
}
