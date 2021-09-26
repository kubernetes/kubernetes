// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package salsa20 implements the Salsa20 stream cipher as specified in https://cr.yp.to/snuffle/spec.pdf.

Salsa20 differs from many other stream ciphers in that it is message orientated
rather than byte orientated. Keystream blocks are not preserved between calls,
therefore each side must encrypt/decrypt data with the same segmentation.

Another aspect of this difference is that part of the counter is exposed as
a nonce in each call. Encrypting two different messages with the same (key,
nonce) pair leads to trivial plaintext recovery. This is analogous to
encrypting two different messages with the same key with a traditional stream
cipher.

This package also implements XSalsa20: a version of Salsa20 with a 24-byte
nonce as specified in https://cr.yp.to/snuffle/xsalsa-20081128.pdf. Simply
passing a 24-byte slice as the nonce triggers XSalsa20.
*/
package salsa20 // import "golang.org/x/crypto/salsa20"

// TODO(agl): implement XORKeyStream12 and XORKeyStream8 - the reduced round variants of Salsa20.

import (
	"golang.org/x/crypto/internal/subtle"
	"golang.org/x/crypto/salsa20/salsa"
)

// XORKeyStream crypts bytes from in to out using the given key and nonce.
// In and out must overlap entirely or not at all. Nonce must
// be either 8 or 24 bytes long.
func XORKeyStream(out, in []byte, nonce []byte, key *[32]byte) {
	if len(out) < len(in) {
		panic("salsa20: output smaller than input")
	}
	if subtle.InexactOverlap(out[:len(in)], in) {
		panic("salsa20: invalid buffer overlap")
	}

	var subNonce [16]byte

	if len(nonce) == 24 {
		var subKey [32]byte
		var hNonce [16]byte
		copy(hNonce[:], nonce[:16])
		salsa.HSalsa20(&subKey, &hNonce, key, &salsa.Sigma)
		copy(subNonce[:], nonce[16:])
		key = &subKey
	} else if len(nonce) == 8 {
		copy(subNonce[:], nonce[:])
	} else {
		panic("salsa20: nonce must be 8 or 24 bytes")
	}

	salsa.XORKeyStream(out, in, &subNonce, key)
}
