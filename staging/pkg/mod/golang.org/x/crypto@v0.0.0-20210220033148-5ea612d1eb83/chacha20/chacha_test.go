// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha20

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math/rand"
	"testing"
)

func _() {
	// Assert that bufSize is a multiple of blockSize.
	var b [1]byte
	_ = b[bufSize%blockSize]
}

func hexDecode(s string) []byte {
	ss, err := hex.DecodeString(s)
	if err != nil {
		panic(fmt.Sprintf("cannot decode input %#v: %v", s, err))
	}
	return ss
}

// Run the test cases with the input and output in different buffers.
func TestNoOverlap(t *testing.T) {
	for _, c := range testVectors {
		s, _ := NewUnauthenticatedCipher(hexDecode(c.key), hexDecode(c.nonce))
		input := hexDecode(c.input)
		output := make([]byte, len(input))
		s.XORKeyStream(output, input)
		got := hex.EncodeToString(output)
		if got != c.output {
			t.Errorf("length=%v: got %#v, want %#v", len(input), got, c.output)
		}
	}
}

// Run the test cases with the input and output overlapping entirely.
func TestOverlap(t *testing.T) {
	for _, c := range testVectors {
		s, _ := NewUnauthenticatedCipher(hexDecode(c.key), hexDecode(c.nonce))
		data := hexDecode(c.input)
		s.XORKeyStream(data, data)
		got := hex.EncodeToString(data)
		if got != c.output {
			t.Errorf("length=%v: got %#v, want %#v", len(data), got, c.output)
		}
	}
}

// Run the test cases with various source and destination offsets.
func TestUnaligned(t *testing.T) {
	const max = 8 // max offset (+1) to test
	for _, c := range testVectors {
		data := hexDecode(c.input)
		input := make([]byte, len(data)+max)
		output := make([]byte, len(data)+max)
		for i := 0; i < max; i++ { // input offsets
			for j := 0; j < max; j++ { // output offsets
				s, _ := NewUnauthenticatedCipher(hexDecode(c.key), hexDecode(c.nonce))

				input := input[i : i+len(data)]
				output := output[j : j+len(data)]

				copy(input, data)
				s.XORKeyStream(output, input)
				got := hex.EncodeToString(output)
				if got != c.output {
					t.Errorf("length=%v: got %#v, want %#v", len(data), got, c.output)
				}
			}
		}
	}
}

// Run the test cases by calling XORKeyStream multiple times.
func TestStep(t *testing.T) {
	// wide range of step sizes to try and hit edge cases
	steps := [...]int{1, 3, 4, 7, 8, 17, 24, 30, 64, 256}
	rnd := rand.New(rand.NewSource(123))
	for _, c := range testVectors {
		s, _ := NewUnauthenticatedCipher(hexDecode(c.key), hexDecode(c.nonce))
		input := hexDecode(c.input)
		output := make([]byte, len(input))

		// step through the buffers
		i, step := 0, steps[rnd.Intn(len(steps))]
		for i+step < len(input) {
			s.XORKeyStream(output[i:i+step], input[i:i+step])
			if i+step < len(input) && output[i+step] != 0 {
				t.Errorf("length=%v, i=%v, step=%v: output overwritten", len(input), i, step)
			}
			i += step
			step = steps[rnd.Intn(len(steps))]
		}
		// finish the encryption
		s.XORKeyStream(output[i:], input[i:])
		// ensure we tolerate a call with an empty input
		s.XORKeyStream(output[len(output):], input[len(input):])

		got := hex.EncodeToString(output)
		if got != c.output {
			t.Errorf("length=%v: got %#v, want %#v", len(input), got, c.output)
		}
	}
}

func TestSetCounter(t *testing.T) {
	newCipher := func() *Cipher {
		s, _ := NewUnauthenticatedCipher(make([]byte, KeySize), make([]byte, NonceSize))
		return s
	}
	s := newCipher()
	src := bytes.Repeat([]byte("test"), 32) // two 64-byte blocks
	dst1 := make([]byte, len(src))
	s.XORKeyStream(dst1, src)
	// advance counter to 1 and xor second block
	s = newCipher()
	s.SetCounter(1)
	dst2 := make([]byte, len(src))
	s.XORKeyStream(dst2[64:], src[64:])
	if !bytes.Equal(dst1[64:], dst2[64:]) {
		t.Error("failed to produce identical output using SetCounter")
	}

	// test again with unaligned blocks; SetCounter should reset the buffer
	s = newCipher()
	s.XORKeyStream(dst1[:70], src[:70])
	s = newCipher()
	s.XORKeyStream([]byte{0}, []byte{0})
	s.SetCounter(1)
	s.XORKeyStream(dst2[64:70], src[64:70])
	if !bytes.Equal(dst1[64:70], dst2[64:70]) {
		t.Error("SetCounter did not reset buffer")
	}

	// advancing to a lower counter value should cause a panic
	panics := func(fn func()) (p bool) {
		defer func() { p = recover() != nil }()
		fn()
		return
	}
	if !panics(func() { s.SetCounter(0) }) {
		t.Error("counter decreasing should trigger a panic")
	}
}

func TestLastBlock(t *testing.T) {
	panics := func(fn func()) (p bool) {
		defer func() { p = recover() != nil }()
		fn()
		return
	}

	checkLastBlock := func(b []byte) {
		t.Helper()
		// Hardcoded result to check all implementations generate the same output.
		lastBlock := "ace4cd09e294d1912d4ad205d06f95d9c2f2bfcf453e8753f128765b62215f4d" +
			"92c74f2f626c6a640c0b1284d839ec81f1696281dafc3e684593937023b58b1d"
		if got := hex.EncodeToString(b); got != lastBlock {
			t.Errorf("wrong output for the last block, got %q, want %q", got, lastBlock)
		}
	}

	// setting the counter to 0xffffffff and crypting multiple blocks should
	// trigger a panic
	s, _ := NewUnauthenticatedCipher(make([]byte, KeySize), make([]byte, NonceSize))
	s.SetCounter(0xffffffff)
	blocks := make([]byte, blockSize*2)
	if !panics(func() { s.XORKeyStream(blocks, blocks) }) {
		t.Error("crypting multiple blocks should trigger a panic")
	}

	// setting the counter to 0xffffffff - 1 and crypting two blocks should not
	// trigger a panic
	s, _ = NewUnauthenticatedCipher(make([]byte, KeySize), make([]byte, NonceSize))
	s.SetCounter(0xffffffff - 1)
	if panics(func() { s.XORKeyStream(blocks, blocks) }) {
		t.Error("crypting the last blocks should not trigger a panic")
	}
	checkLastBlock(blocks[blockSize:])
	// once all the keystream is spent, setting the counter should panic
	if !panics(func() { s.SetCounter(0xffffffff) }) {
		t.Error("setting the counter after overflow should trigger a panic")
	}
	// crypting a subsequent block *should* panic
	block := make([]byte, blockSize)
	if !panics(func() { s.XORKeyStream(block, block) }) {
		t.Error("crypting after overflow should trigger a panic")
	}

	// if we crypt less than a full block, we should be able to crypt the rest
	// in a subsequent call without panicking
	s, _ = NewUnauthenticatedCipher(make([]byte, KeySize), make([]byte, NonceSize))
	s.SetCounter(0xffffffff)
	if panics(func() { s.XORKeyStream(block[:7], block[:7]) }) {
		t.Error("crypting part of the last block should not trigger a panic")
	}
	if panics(func() { s.XORKeyStream(block[7:], block[7:]) }) {
		t.Error("crypting part of the last block should not trigger a panic")
	}
	checkLastBlock(block)
	// as before, a third call should trigger a panic because all keystream is spent
	if !panics(func() { s.XORKeyStream(block[:1], block[:1]) }) {
		t.Error("crypting after overflow should trigger a panic")
	}
}

func benchmarkChaCha20(b *testing.B, step, count int) {
	tot := step * count
	src := make([]byte, tot)
	dst := make([]byte, tot)
	key := make([]byte, KeySize)
	nonce := make([]byte, NonceSize)
	b.SetBytes(int64(tot))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, _ := NewUnauthenticatedCipher(key, nonce)
		for i := 0; i < tot; i += step {
			c.XORKeyStream(dst[i:], src[i:i+step])
		}
	}
}

func BenchmarkChaCha20(b *testing.B) {
	b.Run("64", func(b *testing.B) {
		benchmarkChaCha20(b, 64, 1)
	})
	b.Run("256", func(b *testing.B) {
		benchmarkChaCha20(b, 256, 1)
	})
	b.Run("10x25", func(b *testing.B) {
		benchmarkChaCha20(b, 10, 25)
	})
	b.Run("4096", func(b *testing.B) {
		benchmarkChaCha20(b, 4096, 1)
	})
	b.Run("100x40", func(b *testing.B) {
		benchmarkChaCha20(b, 100, 40)
	})
	b.Run("65536", func(b *testing.B) {
		benchmarkChaCha20(b, 65536, 1)
	})
	b.Run("1000x65", func(b *testing.B) {
		benchmarkChaCha20(b, 1000, 65)
	})
}

func TestHChaCha20(t *testing.T) {
	// See draft-irtf-cfrg-xchacha-00, Section 2.2.1.
	key := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
		0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f}
	nonce := []byte{0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a,
		0x00, 0x00, 0x00, 0x00, 0x31, 0x41, 0x59, 0x27}
	expected := []byte{0x82, 0x41, 0x3b, 0x42, 0x27, 0xb2, 0x7b, 0xfe,
		0xd3, 0x0e, 0x42, 0x50, 0x8a, 0x87, 0x7d, 0x73,
		0xa0, 0xf9, 0xe4, 0xd5, 0x8a, 0x74, 0xa8, 0x53,
		0xc1, 0x2e, 0xc4, 0x13, 0x26, 0xd3, 0xec, 0xdc,
	}
	result, err := HChaCha20(key[:], nonce[:])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(expected, result) {
		t.Errorf("want %x, got %x", expected, result)
	}
}
