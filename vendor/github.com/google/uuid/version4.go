// Copyright 2016 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import "io"

// New creates a new random UUID or panics.  New is equivalent to
// the expression
//
//    uuid.Must(uuid.NewRandom())
func New() UUID {
	return Must(NewRandom())
}

// NewString creates a new random UUID and returns it as a string or panics.
// NewString is equivalent to the expression
//
//    uuid.New().String()
func NewString() string {
	return Must(NewRandom()).String()
}

// NewRandom returns a Random (Version 4) UUID.
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
func NewRandom() (UUID, error) {
	return NewRandomFromReader(rander)
}

// NewRandomFromReader returns a UUID based on bytes read from a given io.Reader.
func NewRandomFromReader(r io.Reader) (UUID, error) {
	var uuid UUID
	_, err := io.ReadFull(r, uuid[:])
	if err != nil {
		return Nil, err
	}
	uuid[6] = (uuid[6] & 0x0f) | 0x40 // Version 4
	uuid[8] = (uuid[8] & 0x3f) | 0x80 // Variant is 10
	return uuid, nil
}
