// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sign

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

var testSignedMessage, _ = hex.DecodeString("26a0a47f733d02ddb74589b6cbd6f64a7dab1947db79395a1a9e00e4c902c0f185b119897b89b248d16bab4ea781b5a3798d25c2984aec833dddab57e0891e0d68656c6c6f20776f726c64")
var testMessage = testSignedMessage[Overhead:]
var testPublicKey [32]byte
var testPrivateKey = [64]byte{
	0x98, 0x3c, 0x6a, 0xa6, 0x21, 0xcc, 0xbb, 0xb2, 0xa7, 0xe8, 0x97, 0x94, 0xde, 0x5f, 0xf8, 0x11,
	0x8a, 0xf3, 0x33, 0x1a, 0x03, 0x5c, 0x43, 0x99, 0x03, 0x13, 0x2d, 0xd7, 0xb4, 0xc4, 0x8b, 0xb0,
	0xf6, 0x33, 0x20, 0xa3, 0x34, 0x8b, 0x7b, 0xe2, 0xfe, 0xb4, 0xe7, 0x3a, 0x54, 0x08, 0x2d, 0xd7,
	0x0c, 0xb7, 0xc0, 0xe3, 0xbf, 0x62, 0x6c, 0x55, 0xf0, 0x33, 0x28, 0x52, 0xf8, 0x48, 0x7d, 0xfd,
}

func init() {
	copy(testPublicKey[:], testPrivateKey[32:])
}

func TestSign(t *testing.T) {
	signedMessage := Sign(nil, testMessage, &testPrivateKey)
	if !bytes.Equal(signedMessage, testSignedMessage) {
		t.Fatalf("signed message did not match, got\n%x\n, expected\n%x", signedMessage, testSignedMessage)
	}
}

func TestOpen(t *testing.T) {
	message, ok := Open(nil, testSignedMessage, &testPublicKey)
	if !ok {
		t.Fatalf("valid signed message not successfully verified")
	}
	if !bytes.Equal(message, testMessage) {
		t.Fatalf("message did not match, got\n%x\n, expected\n%x", message, testMessage)
	}
	_, ok = Open(nil, testSignedMessage[1:], &testPublicKey)
	if ok {
		t.Fatalf("invalid signed message successfully verified")
	}

	badMessage := make([]byte, len(testSignedMessage))
	copy(badMessage, testSignedMessage)
	badMessage[5] ^= 1
	if _, ok := Open(nil, badMessage, &testPublicKey); ok {
		t.Fatalf("Open succeeded with a corrupt message")
	}

	var badPublicKey [32]byte
	copy(badPublicKey[:], testPublicKey[:])
	badPublicKey[5] ^= 1
	if _, ok := Open(nil, testSignedMessage, &badPublicKey); ok {
		t.Fatalf("Open succeeded with a corrupt public key")
	}
}

func TestGenerateSignOpen(t *testing.T) {
	publicKey, privateKey, _ := GenerateKey(rand.Reader)
	signedMessage := Sign(nil, testMessage, privateKey)
	message, ok := Open(nil, signedMessage, publicKey)
	if !ok {
		t.Fatalf("failed to verify signed message")
	}

	if !bytes.Equal(message, testMessage) {
		t.Fatalf("verified message does not match signed messge, got\n%x\n, expected\n%x", message, testMessage)
	}
}
