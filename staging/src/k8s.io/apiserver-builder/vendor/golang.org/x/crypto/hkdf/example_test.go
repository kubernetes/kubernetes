// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hkdf_test

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"golang.org/x/crypto/hkdf"
	"io"
)

// Usage example that expands one master key into three other cryptographically
// secure keys.
func Example_usage() {
	// Underlying hash function to use
	hash := sha256.New

	// Cryptographically secure master key.
	master := []byte{0x00, 0x01, 0x02, 0x03} // i.e. NOT this.

	// Non secret salt, optional (can be nil)
	// Recommended: hash-length sized random
	salt := make([]byte, hash().Size())
	n, err := io.ReadFull(rand.Reader, salt)
	if n != len(salt) || err != nil {
		fmt.Println("error:", err)
		return
	}

	// Non secret context specific info, optional (can be nil).
	// Note, independent from the master key.
	info := []byte{0x03, 0x14, 0x15, 0x92, 0x65}

	// Create the key derivation function
	hkdf := hkdf.New(hash, master, salt, info)

	// Generate the required keys
	keys := make([][]byte, 3)
	for i := 0; i < len(keys); i++ {
		keys[i] = make([]byte, 24)
		n, err := io.ReadFull(hkdf, keys[i])
		if n != len(keys[i]) || err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	// Keys should contain 192 bit random keys
	for i := 1; i <= len(keys); i++ {
		fmt.Printf("Key #%d: %v\n", i, !bytes.Equal(keys[i-1], make([]byte, 24)))
	}

	// Output:
	// Key #1: true
	// Key #2: true
	// Key #3: true
}
