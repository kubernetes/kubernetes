// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tea implements the TEA algorithm, as defined in Needham and
// Wheeler's 1994 technical report, “TEA, a Tiny Encryption Algorithm”. See
// http://www.cix.co.uk/~klockstone/tea.pdf for details.

package tea

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"
)

const (
	// BlockSize is the size of a TEA block, in bytes.
	BlockSize = 8

	// KeySize is the size of a TEA key, in bytes.
	KeySize = 16

	// delta is the TEA key schedule constant.
	delta = 0x9e3779b9

	// numRounds is the standard number of rounds in TEA.
	numRounds = 64
)

// tea is an instance of the TEA cipher with a particular key.
type tea struct {
	key    [16]byte
	rounds int
}

// NewCipher returns an instance of the TEA cipher with the standard number of
// rounds. The key argument must be 16 bytes long.
func NewCipher(key []byte) (cipher.Block, error) {
	return NewCipherWithRounds(key, numRounds)
}

// NewCipherWithRounds returns an instance of the TEA cipher with a given
// number of rounds, which must be even. The key argument must be 16 bytes
// long.
func NewCipherWithRounds(key []byte, rounds int) (cipher.Block, error) {
	if len(key) != 16 {
		return nil, errors.New("tea: incorrect key size")
	}

	if rounds&1 != 0 {
		return nil, errors.New("tea: odd number of rounds specified")
	}

	c := &tea{
		rounds: rounds,
	}
	copy(c.key[:], key)

	return c, nil
}

// BlockSize returns the TEA block size, which is eight bytes. It is necessary
// to satisfy the Block interface in the package "crypto/cipher".
func (*tea) BlockSize() int {
	return BlockSize
}

// Encrypt encrypts the 8 byte buffer src using the key in t and stores the
// result in dst. Note that for amounts of data larger than a block, it is not
// safe to just call Encrypt on successive blocks; instead, use an encryption
// mode like CBC (see crypto/cipher/cbc.go).
func (t *tea) Encrypt(dst, src []byte) {
	e := binary.BigEndian
	v0, v1 := e.Uint32(src), e.Uint32(src[4:])
	k0, k1, k2, k3 := e.Uint32(t.key[0:]), e.Uint32(t.key[4:]), e.Uint32(t.key[8:]), e.Uint32(t.key[12:])

	sum := uint32(0)
	delta := uint32(delta)

	for i := 0; i < t.rounds/2; i++ {
		sum += delta
		v0 += ((v1 << 4) + k0) ^ (v1 + sum) ^ ((v1 >> 5) + k1)
		v1 += ((v0 << 4) + k2) ^ (v0 + sum) ^ ((v0 >> 5) + k3)
	}

	e.PutUint32(dst, v0)
	e.PutUint32(dst[4:], v1)
}

// Decrypt decrypts the 8 byte buffer src using the key in t and stores the
// result in dst.
func (t *tea) Decrypt(dst, src []byte) {
	e := binary.BigEndian
	v0, v1 := e.Uint32(src), e.Uint32(src[4:])
	k0, k1, k2, k3 := e.Uint32(t.key[0:]), e.Uint32(t.key[4:]), e.Uint32(t.key[8:]), e.Uint32(t.key[12:])

	delta := uint32(delta)
	sum := delta * uint32(t.rounds/2) // in general, sum = delta * n

	for i := 0; i < t.rounds/2; i++ {
		v1 -= ((v0 << 4) + k2) ^ (v0 + sum) ^ ((v0 >> 5) + k3)
		v0 -= ((v1 << 4) + k0) ^ (v1 + sum) ^ ((v1 >> 5) + k1)
		sum -= delta
	}

	e.PutUint32(dst, v0)
	e.PutUint32(dst[4:], v1)
}
