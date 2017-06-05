// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tea

import (
	"bytes"
	"testing"
)

// A sample test key for when we just want to initialize a cipher
var testKey = []byte{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF}

// Test that the block size for tea is correct
func TestBlocksize(t *testing.T) {
	c, err := NewCipher(testKey)
	if err != nil {
		t.Fatalf("NewCipher returned error: %s", err)
	}

	if result := c.BlockSize(); result != BlockSize {
		t.Errorf("cipher.BlockSize returned %d, but expected %d", result, BlockSize)
	}
}

// Test that invalid key sizes return an error
func TestInvalidKeySize(t *testing.T) {
	var key [KeySize + 1]byte

	if _, err := NewCipher(key[:]); err == nil {
		t.Errorf("invalid key size %d didn't result in an error.", len(key))
	}

	if _, err := NewCipher(key[:KeySize-1]); err == nil {
		t.Errorf("invalid key size %d didn't result in an error.", KeySize-1)
	}
}

// Test Vectors
type teaTest struct {
	rounds     int
	key        []byte
	plaintext  []byte
	ciphertext []byte
}

var teaTests = []teaTest{
	// These were sourced from https://github.com/froydnj/ironclad/blob/master/testing/test-vectors/tea.testvec
	{
		numRounds,
		[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		[]byte{0x41, 0xea, 0x3a, 0x0a, 0x94, 0xba, 0xa9, 0x40},
	},
	{
		numRounds,
		[]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		[]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		[]byte{0x31, 0x9b, 0xbe, 0xfb, 0x01, 0x6a, 0xbd, 0xb2},
	},
	{
		16,
		[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		[]byte{0xed, 0x28, 0x5d, 0xa1, 0x45, 0x5b, 0x33, 0xc1},
	},
}

// Test encryption
func TestCipherEncrypt(t *testing.T) {
	// Test encryption with standard 64 rounds
	for i, test := range teaTests {
		c, err := NewCipherWithRounds(test.key, test.rounds)
		if err != nil {
			t.Fatalf("#%d: NewCipher returned error: %s", i, err)
		}

		var ciphertext [BlockSize]byte
		c.Encrypt(ciphertext[:], test.plaintext)

		if !bytes.Equal(ciphertext[:], test.ciphertext) {
			t.Errorf("#%d: incorrect ciphertext. Got %x, wanted %x", i, ciphertext, test.ciphertext)
		}

		var plaintext2 [BlockSize]byte
		c.Decrypt(plaintext2[:], ciphertext[:])

		if !bytes.Equal(plaintext2[:], test.plaintext) {
			t.Errorf("#%d: incorrect plaintext. Got %x, wanted %x", i, plaintext2, test.plaintext)
		}
	}
}
