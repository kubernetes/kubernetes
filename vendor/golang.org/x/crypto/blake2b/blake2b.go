// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package blake2b implements the BLAKE2b hash algorithm defined by RFC 7693
// and the extendable output function (XOF) BLAKE2Xb.
//
// For a detailed specification of BLAKE2b see https://blake2.net/blake2.pdf
// and for BLAKE2Xb see https://blake2.net/blake2x.pdf
//
// If you aren't sure which function you need, use BLAKE2b (Sum512 or New512).
// If you need a secret-key MAC (message authentication code), use the New512
// function with a non-nil key.
//
// BLAKE2X is a construction to compute hash values larger than 64 bytes. It
// can produce hash values between 0 and 4 GiB.
package blake2b

import (
	"encoding/binary"
	"errors"
	"hash"
)

const (
	// The blocksize of BLAKE2b in bytes.
	BlockSize = 128
	// The hash size of BLAKE2b-512 in bytes.
	Size = 64
	// The hash size of BLAKE2b-384 in bytes.
	Size384 = 48
	// The hash size of BLAKE2b-256 in bytes.
	Size256 = 32
)

var (
	useAVX2 bool
	useAVX  bool
	useSSE4 bool
)

var errKeySize = errors.New("blake2b: invalid key size")

var iv = [8]uint64{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
}

// Sum512 returns the BLAKE2b-512 checksum of the data.
func Sum512(data []byte) [Size]byte {
	var sum [Size]byte
	checkSum(&sum, Size, data)
	return sum
}

// Sum384 returns the BLAKE2b-384 checksum of the data.
func Sum384(data []byte) [Size384]byte {
	var sum [Size]byte
	var sum384 [Size384]byte
	checkSum(&sum, Size384, data)
	copy(sum384[:], sum[:Size384])
	return sum384
}

// Sum256 returns the BLAKE2b-256 checksum of the data.
func Sum256(data []byte) [Size256]byte {
	var sum [Size]byte
	var sum256 [Size256]byte
	checkSum(&sum, Size256, data)
	copy(sum256[:], sum[:Size256])
	return sum256
}

// New512 returns a new hash.Hash computing the BLAKE2b-512 checksum. A non-nil
// key turns the hash into a MAC. The key must between zero and 64 bytes long.
func New512(key []byte) (hash.Hash, error) { return newDigest(Size, key) }

// New384 returns a new hash.Hash computing the BLAKE2b-384 checksum. A non-nil
// key turns the hash into a MAC. The key must between zero and 64 bytes long.
func New384(key []byte) (hash.Hash, error) { return newDigest(Size384, key) }

// New256 returns a new hash.Hash computing the BLAKE2b-256 checksum. A non-nil
// key turns the hash into a MAC. The key must between zero and 64 bytes long.
func New256(key []byte) (hash.Hash, error) { return newDigest(Size256, key) }

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
	h := iv
	h[0] ^= uint64(hashSize) | (1 << 16) | (1 << 24)
	var c [2]uint64

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
	remaining := uint64(BlockSize - offset)
	if c[0] < remaining {
		c[1]--
	}
	c[0] -= remaining

	hashBlocks(&h, &c, 0xFFFFFFFFFFFFFFFF, block[:])

	for i, v := range h[:(hashSize+7)/8] {
		binary.LittleEndian.PutUint64(sum[8*i:], v)
	}
}

type digest struct {
	h      [8]uint64
	c      [2]uint64
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
	d.h[0] ^= uint64(d.size) | (uint64(d.keyLen) << 8) | (1 << 16) | (1 << 24)
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

	if len(p) > 0 {
		d.offset += copy(d.block[:], p)
	}

	return
}

func (d *digest) Sum(sum []byte) []byte {
	var hash [Size]byte
	d.finalize(&hash)
	return append(sum, hash[:d.size]...)
}

func (d *digest) finalize(hash *[Size]byte) {
	var block [BlockSize]byte
	copy(block[:], d.block[:d.offset])
	remaining := uint64(BlockSize - d.offset)

	c := d.c
	if c[0] < remaining {
		c[1]--
	}
	c[0] -= remaining

	h := d.h
	hashBlocks(&h, &c, 0xFFFFFFFFFFFFFFFF, block[:])

	for i, v := range h {
		binary.LittleEndian.PutUint64(hash[8*i:], v)
	}
}
