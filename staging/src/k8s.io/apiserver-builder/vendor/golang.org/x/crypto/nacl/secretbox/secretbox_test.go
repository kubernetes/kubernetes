// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package secretbox

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func TestSealOpen(t *testing.T) {
	var key [32]byte
	var nonce [24]byte

	rand.Reader.Read(key[:])
	rand.Reader.Read(nonce[:])

	var box, opened []byte

	for msgLen := 0; msgLen < 128; msgLen += 17 {
		message := make([]byte, msgLen)
		rand.Reader.Read(message)

		box = Seal(box[:0], message, &nonce, &key)
		var ok bool
		opened, ok = Open(opened[:0], box, &nonce, &key)
		if !ok {
			t.Errorf("%d: failed to open box", msgLen)
			continue
		}

		if !bytes.Equal(opened, message) {
			t.Errorf("%d: got %x, expected %x", msgLen, opened, message)
			continue
		}
	}

	for i := range box {
		box[i] ^= 0x20
		_, ok := Open(opened[:0], box, &nonce, &key)
		if ok {
			t.Errorf("box was opened after corrupting byte %d", i)
		}
		box[i] ^= 0x20
	}
}

func TestSecretBox(t *testing.T) {
	var key [32]byte
	var nonce [24]byte
	var message [64]byte

	for i := range key[:] {
		key[i] = 1
	}
	for i := range nonce[:] {
		nonce[i] = 2
	}
	for i := range message[:] {
		message[i] = 3
	}

	box := Seal(nil, message[:], &nonce, &key)
	// expected was generated using the C implementation of NaCl.
	expected, _ := hex.DecodeString("8442bc313f4626f1359e3b50122b6ce6fe66ddfe7d39d14e637eb4fd5b45beadab55198df6ab5368439792a23c87db70acb6156dc5ef957ac04f6276cf6093b84be77ff0849cc33e34b7254d5a8f65ad")

	if !bytes.Equal(box, expected) {
		t.Fatalf("box didn't match, got\n%x\n, expected\n%x", box, expected)
	}
}

func TestAppend(t *testing.T) {
	var key [32]byte
	var nonce [24]byte
	var message [8]byte

	out := make([]byte, 4)
	box := Seal(out, message[:], &nonce, &key)
	if !bytes.Equal(box[:4], out[:4]) {
		t.Fatalf("Seal didn't correctly append")
	}

	out = make([]byte, 4, 100)
	box = Seal(out, message[:], &nonce, &key)
	if !bytes.Equal(box[:4], out[:4]) {
		t.Fatalf("Seal didn't correctly append with sufficient capacity.")
	}
}
