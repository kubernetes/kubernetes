// Copyright 2011 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"io"

	guuid "github.com/google/uuid"
)

// Array is a pass-by-value UUID that can be used as an effecient key in a map.
type Array [16]byte

// UUID converts uuid into a slice.
func (uuid Array) UUID() UUID {
	return uuid[:]
}

// String returns the string representation of uuid,
// xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
func (uuid Array) String() string {
	return guuid.UUID(uuid).String()
}

// A UUID is a 128 bit (16 byte) Universal Unique IDentifier as defined in RFC
// 4122.
type UUID []byte

// A Version represents a UUIDs version.
type Version = guuid.Version

// A Variant represents a UUIDs variant.
type Variant = guuid.Variant

// Constants returned by Variant.
const (
	Invalid   = guuid.Invalid   // Invalid UUID
	RFC4122   = guuid.RFC4122   // The variant specified in RFC4122
	Reserved  = guuid.Reserved  // Reserved, NCS backward compatibility.
	Microsoft = guuid.Microsoft // Reserved, Microsoft Corporation backward compatibility.
	Future    = guuid.Future    // Reserved for future definition.
)

var rander = rand.Reader // random function

// New returns a new random (version 4) UUID as a string.  It is a convenience
// function for NewRandom().String().
func New() string {
	return NewRandom().String()
}

// Parse decodes s into a UUID or returns nil. See github.com/google/uuid for
// the formats parsed.
func Parse(s string) UUID {
	gu, err := guuid.Parse(s)
	if err == nil {
		return gu[:]
	}
	return nil
}

// ParseBytes is like Parse, except it parses a byte slice instead of a string.
func ParseBytes(b []byte) (UUID, error) {
	gu, err := guuid.ParseBytes(b)
	if err == nil {
		return gu[:], nil
	}
	return nil, err
}

// Equal returns true if uuid1 and uuid2 are equal.
func Equal(uuid1, uuid2 UUID) bool {
	return bytes.Equal(uuid1, uuid2)
}

// Array returns an array representation of uuid that can be used as a map key.
// Array panics if uuid is not valid.
func (uuid UUID) Array() Array {
	if len(uuid) != 16 {
		panic("invalid uuid")
	}
	var a Array
	copy(a[:], uuid)
	return a
}

// String returns the string form of uuid, xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
// , or "" if uuid is invalid.
func (uuid UUID) String() string {
	if len(uuid) != 16 {
		return ""
	}
	var buf [36]byte
	encodeHex(buf[:], uuid)
	return string(buf[:])
}

// URN returns the RFC 2141 URN form of uuid,
// urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx,  or "" if uuid is invalid.
func (uuid UUID) URN() string {
	if len(uuid) != 16 {
		return ""
	}
	var buf [36 + 9]byte
	copy(buf[:], "urn:uuid:")
	encodeHex(buf[9:], uuid)
	return string(buf[:])
}

func encodeHex(dst []byte, uuid UUID) {
	hex.Encode(dst[:], uuid[:4])
	dst[8] = '-'
	hex.Encode(dst[9:13], uuid[4:6])
	dst[13] = '-'
	hex.Encode(dst[14:18], uuid[6:8])
	dst[18] = '-'
	hex.Encode(dst[19:23], uuid[8:10])
	dst[23] = '-'
	hex.Encode(dst[24:], uuid[10:])
}

// Variant returns the variant encoded in uuid.  It returns Invalid if
// uuid is invalid.
func (uuid UUID) Variant() Variant {
	if len(uuid) != 16 {
		return Invalid
	}
	switch {
	case (uuid[8] & 0xc0) == 0x80:
		return RFC4122
	case (uuid[8] & 0xe0) == 0xc0:
		return Microsoft
	case (uuid[8] & 0xe0) == 0xe0:
		return Future
	default:
		return Reserved
	}
}

// Version returns the version of uuid.  It returns false if uuid is not
// valid.
func (uuid UUID) Version() (Version, bool) {
	if len(uuid) != 16 {
		return 0, false
	}
	return Version(uuid[6] >> 4), true
}

// SetRand sets the random number generator to r, which implements io.Reader.
// If r.Read returns an error when the package requests random data then
// a panic will be issued.
//
// Calling SetRand with nil sets the random number generator to the default
// generator.
func SetRand(r io.Reader) {
	guuid.SetRand(r)
}
