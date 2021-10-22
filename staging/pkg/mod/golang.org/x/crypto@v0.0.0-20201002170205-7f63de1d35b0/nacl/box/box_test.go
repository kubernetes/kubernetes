// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package box

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"testing"

	"golang.org/x/crypto/curve25519"
)

func TestSealOpen(t *testing.T) {
	publicKey1, privateKey1, _ := GenerateKey(rand.Reader)
	publicKey2, privateKey2, _ := GenerateKey(rand.Reader)

	if *privateKey1 == *privateKey2 {
		t.Fatalf("private keys are equal!")
	}
	if *publicKey1 == *publicKey2 {
		t.Fatalf("public keys are equal!")
	}
	message := []byte("test message")
	var nonce [24]byte

	box := Seal(nil, message, &nonce, publicKey1, privateKey2)
	opened, ok := Open(nil, box, &nonce, publicKey2, privateKey1)
	if !ok {
		t.Fatalf("failed to open box")
	}

	if !bytes.Equal(opened, message) {
		t.Fatalf("got %x, want %x", opened, message)
	}

	for i := range box {
		box[i] ^= 0x40
		_, ok := Open(nil, box, &nonce, publicKey2, privateKey1)
		if ok {
			t.Fatalf("opened box with byte %d corrupted", i)
		}
		box[i] ^= 0x40
	}
}

func TestBox(t *testing.T) {
	var privateKey1, privateKey2 [32]byte
	for i := range privateKey1[:] {
		privateKey1[i] = 1
	}
	for i := range privateKey2[:] {
		privateKey2[i] = 2
	}

	var publicKey1 [32]byte
	curve25519.ScalarBaseMult(&publicKey1, &privateKey1)
	var message [64]byte
	for i := range message[:] {
		message[i] = 3
	}

	var nonce [24]byte
	for i := range nonce[:] {
		nonce[i] = 4
	}

	box := Seal(nil, message[:], &nonce, &publicKey1, &privateKey2)

	// expected was generated using the C implementation of NaCl.
	expected, _ := hex.DecodeString("78ea30b19d2341ebbdba54180f821eec265cf86312549bea8a37652a8bb94f07b78a73ed1708085e6ddd0e943bbdeb8755079a37eb31d86163ce241164a47629c0539f330b4914cd135b3855bc2a2dfc")

	if !bytes.Equal(box, expected) {
		t.Fatalf("box didn't match, got\n%x\n, expected\n%x", box, expected)
	}
}

func TestSealOpenAnonymous(t *testing.T) {
	publicKey, privateKey, _ := GenerateKey(rand.Reader)
	message := []byte("test message")

	box, err := SealAnonymous(nil, message, publicKey, nil)
	if err != nil {
		t.Fatalf("Unexpected error sealing %v", err)
	}
	opened, ok := OpenAnonymous(nil, box, publicKey, privateKey)
	if !ok {
		t.Fatalf("failed to open box")
	}

	if !bytes.Equal(opened, message) {
		t.Fatalf("got %x, want %x", opened, message)
	}

	for i := range box {
		box[i] ^= 0x40
		_, ok := OpenAnonymous(nil, box, publicKey, privateKey)
		if ok {
			t.Fatalf("opened box with byte %d corrupted", i)
		}
		box[i] ^= 0x40
	}

	// allocates new slice if out isn't long enough
	out := []byte("hello")
	orig := append([]byte(nil), out...)
	box, err = SealAnonymous(out, message, publicKey, nil)
	if err != nil {
		t.Fatalf("Unexpected error sealing %v", err)
	}
	if !bytes.Equal(out, orig) {
		t.Fatal("expected out to be unchanged")
	}
	if !bytes.HasPrefix(box, orig) {
		t.Fatal("expected out to be coppied to returned slice")
	}
	_, ok = OpenAnonymous(nil, box[len(out):], publicKey, privateKey)
	if !ok {
		t.Fatalf("failed to open box")
	}

	// uses provided slice if it's long enough
	out = append(make([]byte, 0, 1000), []byte("hello")...)
	orig = append([]byte(nil), out...)
	box, err = SealAnonymous(out, message, publicKey, nil)
	if err != nil {
		t.Fatalf("Unexpected error sealing %v", err)
	}
	if !bytes.Equal(out, orig) {
		t.Fatal("expected out to be unchanged")
	}
	if &out[0] != &box[0] {
		t.Fatal("expected box to point to out")
	}
	_, ok = OpenAnonymous(nil, box[len(out):], publicKey, privateKey)
	if !ok {
		t.Fatalf("failed to open box")
	}
}

func TestSealedBox(t *testing.T) {
	var privateKey [32]byte
	for i := range privateKey[:] {
		privateKey[i] = 1
	}

	var publicKey [32]byte
	curve25519.ScalarBaseMult(&publicKey, &privateKey)
	var message [64]byte
	for i := range message[:] {
		message[i] = 3
	}

	fakeRand := bytes.NewReader([]byte{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5})
	box, err := SealAnonymous(nil, message[:], &publicKey, fakeRand)
	if err != nil {
		t.Fatalf("Unexpected error sealing %v", err)
	}

	// expected was generated using the C implementation of libsodium with a
	// random implementation that always returns 5.
	// https://gist.github.com/mastahyeti/942ec3f175448d68fed25018adbce5a7
	expected, _ := hex.DecodeString("50a61409b1ddd0325e9b16b700e719e9772c07000b1bd7786e907c653d20495d2af1697137a53b1b1dfc9befc49b6eeb38f86be720e155eb2be61976d2efb34d67ecd44a6ad634625eb9c288bfc883431a84ab0f5557dfe673aa6f74c19f033e648a947358cfcc606397fa1747d5219a")

	if !bytes.Equal(box, expected) {
		t.Fatalf("box didn't match, got\n%x\n, expected\n%x", box, expected)
	}

	// box was generated using the C implementation of libsodium.
	// https://gist.github.com/mastahyeti/942ec3f175448d68fed25018adbce5a7
	box, _ = hex.DecodeString("3462e0640728247a6f581e3812850d6edc3dcad1ea5d8184c072f62fb65cb357e27ffa8b76f41656bc66a0882c4d359568410665746d27462a700f01e314f382edd7aae9064879b0f8ba7b88866f88f5e4fbd7649c850541877f9f33ebd25d46d9cbcce09b69a9ba07f0eb1d105d4264")
	result, ok := OpenAnonymous(nil, box, &publicKey, &privateKey)
	if !ok {
		t.Fatalf("failed to open box")
	}
	if !bytes.Equal(result, message[:]) {
		t.Fatalf("message didn't match, got\n%x\n, expected\n%x", result, message[:])
	}
}
