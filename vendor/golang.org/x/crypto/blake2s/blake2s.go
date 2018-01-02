// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package blake2s implements the BLAKE2s hash algorithm defined by RFC 7693
// and the extendable output function (XOF) BLAKE2Xs.
//
// For a detailed specification of BLAKE2s see https://blake2.net/blake2.pdf
// and for BLAKE2Xs see https://blake2.net/blake2x.pdf
//
// If you aren't sure which function you need, use BLAKE2s (Sum256 or New256).
// If you need a secret-key MAC (message authentication code), use the New256
// function with a non-nil key.
//
// BLAKE2X is a construction to compute hash values larger than 32 bytes. It
// can produce hash values between 0 and 65535 bytes.
package blake2s // import "golang.org/x/crypto/blake2s"

import (
	"encoding/binary"
	"errors"
	"hash"
)

const (
	// The blocksize of BLAKE2s in bytes.
	BlockSize = 64

	// The hash size of BLAKE2s-256 in bytes.
	Size = 32

	// The hash size of BLAKE2s-128 in bytes.
	Size128 = 16
)

var errKeySize = errors.New("blake2s: invalid key size")

var iv = [8]uint32{
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
}

// Sum256 returns the BLAKE2s-256 checksum of the data.
func Sum256(data []byte) [Size]byte {
	var sum [Size]byte
	checkSum(&sum, Size, data)
	return sum
}

// New256 returns a new hash.Hash computing the BLAKE2s-256 checksum. A non-nil
// key turns the hash into a MAC. The key must between zero and 32 bytes long.
func New256(key []byte) (hash.Hash, error) { return newDigest(Size, key) }

// New128 returns a new hash.Hash computing the BLAKE2s-128 checksum given a
// non-empty key. Note that a 128-bit digest is too small to be secure as a
// cryptographic hash and should only be used as a MAC, thus the key argument
// is not optional.
func New128(key []byte) (hash.Hash, error) {
	if len(key) == 0 {
		return nil, errors.New("blake2s: a key is required for a 128-bit hash")
	}
	return newDigest(Size128, key)
}

func newDigest(hashSize int, key []byte) (*digest, error) {
	if len(key) > Size {
		return nil, errKeySize
	}
	d := &digest{
		size:   hashSize,
		keyLen: len(key),
	}
	copy(d.key[:], key)
	d.Reset()
	return d, nil
}

func checkSum(sum *[Size]byte, hashSize int, data []byte) {
	var (
		h [8]uint32
		c [2]uint32
	)

	h = iv
	h[0] ^= uint32(hashSize) | (1 << 16) | (1 << 24)

	if length := len(data); length > BlockSize {
		n := length &^ (BlockSize - 1)
		if length == n {
			n -= BlockSize
		}
		hashBlocks(&h, &c, 0, data[:n])
		data = data[n:]
	}

	var block [BlockSize]byte
	offset := copy(block[:], data)
	remaining := uint32(BlockSize - offset)

	if c[0] < remaining {
		c[1]--
	}
	c[0] -= remaining

	hashBlocks(&h, &c, 0xFFFFFFFF, block[:])

	for i, v := range h {
		binary.LittleEndian.PutUint32(sum[4*i:], v)
	}
}

type digest struct {
	h      [8]uint32
	c      [2]uint32
	size   int
	block  [BlockSize]byte
	offset int

	key    [BlockSize]byte
	keyLen int
}

func (d *digest) BlockSize() int { return BlockSize }

func (d *digest) Size() int { return d.size }

func (d *digest) Reset() {
	d.h = iv
	d.h[0] ^= uint32(d.size) | (uint32(d.keyLen) << 8) | (1 << 16) | (1 << 24)
	d.offset, d.c[0], d.c[1] = 0, 0, 0
	if d.keyLen > 0 {
		d.block = d.key
		d.offset = BlockSize
	}
}

func (d *digest) Write(p []byte) (n int, err error) {
	n = len(p)

	if d.offset > 0 {
		remaining := BlockSize - d.offset
		if n <= remaining {
			d.offset += copy(d.block[d.offset:], p)
			return
		}
		copy(d.block[d.offset:], p[:remaining])
		hashBlocks(&d.h, &d.c, 0, d.block[:])
		d.offset = 0
		p = p[remaining:]
	}

	if length := len(p); length > BlockSize {
		nn := length &^ (BlockSize - 1)
		if length == nn {
			nn -= BlockSize
		}
		hashBlocks(&d.h, &d.c, 0, p[:nn])
		p = p[nn:]
	}

	d.offset += copy(d.block[:], p)
	return
}

func (d *digest) Sum(sum []byte) []byte {
	var hash [Size]byte
	d.finalize(&hash)
	return append(sum, hash[:d.size]...)
}

func (d *digest) finalize(hash *[Size]byte) {
	var block [BlockSize]byte
	h := d.h
	c := d.c

	copy(block[:], d.block[:d.offset])
	remaining := uint32(BlockSize - d.offset)
	if c[0] < remaining {
		c[1]--
	}
	c[0] -= remaining

	hashBlocks(&h, &c, 0xFFFFFFFF, block[:])
	for i, v := range h {
		binary.LittleEndian.PutUint32(hash[4*i:], v)
	}
}
