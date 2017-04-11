// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import (
	"bytes"
	"testing"
)

func TestThatPBKDFWorksCorrectlyForLongKeys(t *testing.T) {
	cipherInfo := shaWithTripleDESCBC{}

	salt := []byte("\xff\xff\xff\xff\xff\xff\xff\xff")
	password, _ := bmpString("sesame")
	key := cipherInfo.deriveKey(salt, password, 2048)

	if expected := []byte("\x7c\xd9\xfd\x3e\x2b\x3b\xe7\x69\x1a\x44\xe3\xbe\xf0\xf9\xea\x0f\xb9\xb8\x97\xd4\xe3\x25\xd9\xd1"); bytes.Compare(key, expected) != 0 {
		t.Fatalf("expected key '%x', but found '%x'", expected, key)
	}
}

func TestThatPBKDFHandlesLeadingZeros(t *testing.T) {
	// This test triggers a case where I_j (in step 6C) ends up with leading zero
	// byte, meaning that len(Ijb) < v (leading zeros get stripped by big.Int).
	// This was previously causing bug whereby certain inputs would break the
	// derivation and produce the wrong output.
	key := pbkdf(sha1Sum, 20, 64, []byte("\xf3\x7e\x05\xb5\x18\x32\x4b\x4b"), []byte("\x00\x00"), 2048, 1, 24)
	expected := []byte("\x00\xf7\x59\xff\x47\xd1\x4d\xd0\x36\x65\xd5\x94\x3c\xb3\xc4\xa3\x9a\x25\x55\xc0\x2a\xed\x66\xe1")
	if bytes.Compare(key, expected) != 0 {
		t.Fatalf("expected key '%x', but found '%x'", expected, key)
	}
}
