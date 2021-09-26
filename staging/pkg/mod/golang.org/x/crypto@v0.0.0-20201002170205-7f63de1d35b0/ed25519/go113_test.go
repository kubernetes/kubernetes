// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.13

package ed25519_test

import (
	ed25519std "crypto/ed25519"
	"golang.org/x/crypto/ed25519"
	"testing"
)

func TestTypeAlias(t *testing.T) {
	var zero zeroReader
	public, private, _ := ed25519std.GenerateKey(zero)

	message := []byte("test message")
	sig := ed25519.Sign(private, message)
	if !ed25519.Verify(public, message, sig) {
		t.Errorf("valid signature rejected")
	}
}
