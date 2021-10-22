// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sign signs small messages using public-key cryptography.
//
// Sign uses Ed25519 to sign messages. The length of messages is not hidden.
// Messages should be small because:
// 1. The whole message needs to be held in memory to be processed.
// 2. Using large messages pressures implementations on small machines to process
// plaintext without verifying the signature. This is very dangerous, and this API
// discourages it, but a protocol that uses excessive message sizes might present
// some implementations with no other choice.
// 3. Performance may be improved by working with messages that fit into data caches.
// Thus large amounts of data should be chunked so that each message is small.
//
// This package is not interoperable with the current release of NaCl
// (https://nacl.cr.yp.to/sign.html), which does not support Ed25519 yet. However,
// it is compatible with the NaCl fork libsodium (https://www.libsodium.org), as well
// as TweetNaCl (https://tweetnacl.cr.yp.to/).
package sign

import (
	"io"

	"golang.org/x/crypto/ed25519"
	"golang.org/x/crypto/internal/subtle"
)

// Overhead is the number of bytes of overhead when signing a message.
const Overhead = 64

// GenerateKey generates a new public/private key pair suitable for use with
// Sign and Open.
func GenerateKey(rand io.Reader) (publicKey *[32]byte, privateKey *[64]byte, err error) {
	pub, priv, err := ed25519.GenerateKey(rand)
	if err != nil {
		return nil, nil, err
	}
	publicKey, privateKey = new([32]byte), new([64]byte)
	copy((*publicKey)[:], pub)
	copy((*privateKey)[:], priv)
	return publicKey, privateKey, nil
}

// Sign appends a signed copy of message to out, which will be Overhead bytes
// longer than the original and must not overlap it.
func Sign(out, message []byte, privateKey *[64]byte) []byte {
	sig := ed25519.Sign(ed25519.PrivateKey((*privateKey)[:]), message)
	ret, out := sliceForAppend(out, Overhead+len(message))
	if subtle.AnyOverlap(out, message) {
		panic("nacl: invalid buffer overlap")
	}
	copy(out, sig)
	copy(out[Overhead:], message)
	return ret
}

// Open verifies a signed message produced by Sign and appends the message to
// out, which must not overlap the signed message. The output will be Overhead
// bytes smaller than the signed message.
func Open(out, signedMessage []byte, publicKey *[32]byte) ([]byte, bool) {
	if len(signedMessage) < Overhead {
		return nil, false
	}
	if !ed25519.Verify(ed25519.PublicKey((*publicKey)[:]), signedMessage[Overhead:], signedMessage[:Overhead]) {
		return nil, false
	}
	ret, out := sliceForAppend(out, len(signedMessage)-Overhead)
	if subtle.AnyOverlap(out, signedMessage) {
		panic("nacl: invalid buffer overlap")
	}
	copy(out, signedMessage[Overhead:])
	return ret, true
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}
