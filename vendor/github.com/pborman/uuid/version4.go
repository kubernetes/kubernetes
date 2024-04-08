// Copyright 2011 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import guuid "github.com/google/uuid"

// NewRandom returns a Random (Version 4) UUID or panics.
//
// The strength of the UUIDs is based on the strength of the crypto/rand
// package.
//
// A note about uniqueness derived from the UUID Wikipedia entry:
//
//  Randomly generated UUIDs have 122 random bits.  One's annual risk of being
//  hit by a meteorite is estimated to be one chance in 17 billion, that
//  means the probability is about 0.00000000006 (6 × 10−11),
//  equivalent to the odds of creating a few tens of trillions of UUIDs in a
//  year and having one duplicate.
func NewRandom() UUID {
	if gu, err := guuid.NewRandom(); err == nil {
		return UUID(gu[:])
	}
	return nil
}
