// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xz

import (
	"hash"
	"hash/crc32"
	"hash/crc64"
)

// crc32Hash implements the hash.Hash32 interface with Sum returning the
// crc32 value in little-endian encoding.
type crc32Hash struct {
	hash.Hash32
}

// Sum returns the crc32 value as little endian.
func (h crc32Hash) Sum(b []byte) []byte {
	p := make([]byte, 4)
	putUint32LE(p, h.Hash32.Sum32())
	b = append(b, p...)
	return b
}

// newCRC32 returns a CRC-32 hash that returns the 64-bit value in
// little-endian encoding using the IEEE polynomial.
func newCRC32() hash.Hash {
	return crc32Hash{Hash32: crc32.NewIEEE()}
}

// crc64Hash implements the Hash64 interface with Sum returning the
// CRC-64 value in little-endian encoding.
type crc64Hash struct {
	hash.Hash64
}

// Sum returns the CRC-64 value in little-endian encoding.
func (h crc64Hash) Sum(b []byte) []byte {
	p := make([]byte, 8)
	putUint64LE(p, h.Hash64.Sum64())
	b = append(b, p...)
	return b
}

// crc64Table is used to create a CRC-64 hash.
var crc64Table = crc64.MakeTable(crc64.ECMA)

// newCRC64 returns a CRC-64 hash that returns the 64-bit value in
// little-endian encoding using the ECMA polynomial.
func newCRC64() hash.Hash {
	return crc64Hash{Hash64: crc64.New(crc64Table)}
}
