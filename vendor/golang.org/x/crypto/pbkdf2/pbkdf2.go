// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pbkdf2 implements the key derivation function PBKDF2 as defined in
// RFC 8018 (PKCS #5 v2.1).
//
// This package is a wrapper for the PBKDF2 implementation in the
// [crypto/pbkdf2] package. It is [frozen] and is not accepting new features.
//
// [frozen]: https://go.dev/wiki/Frozen
package pbkdf2

import (
	"crypto/pbkdf2"
	"hash"
)

// Key derives a key from the password, salt and iteration count, returning a
// []byte of length keylen that can be used as cryptographic key. The key is
// derived based on the method described as PBKDF2 with the HMAC variant using
// the supplied hash function.
func Key(password, salt []byte, iter, keyLen int, h func() hash.Hash) []byte {
	out, err := pbkdf2.Key(h, string(password), salt, iter, keyLen)
	if err != nil {
		// FIPS 140 enforcement, or an invalid key length.
		panic(err)
	}
	return out
}
