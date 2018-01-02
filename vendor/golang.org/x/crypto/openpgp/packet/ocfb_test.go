// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto/aes"
	"crypto/rand"
	"testing"
)

var commonKey128 = []byte{0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c}

func testOCFB(t *testing.T, resync OCFBResyncOption) {
	block, err := aes.NewCipher(commonKey128)
	if err != nil {
		t.Error(err)
		return
	}

	plaintext := []byte("this is the plaintext, which is long enough to span several blocks.")
	randData := make([]byte, block.BlockSize())
	rand.Reader.Read(randData)
	ocfb, prefix := NewOCFBEncrypter(block, randData, resync)
	ciphertext := make([]byte, len(plaintext))
	ocfb.XORKeyStream(ciphertext, plaintext)

	ocfbdec := NewOCFBDecrypter(block, prefix, resync)
	if ocfbdec == nil {
		t.Errorf("NewOCFBDecrypter failed (resync: %t)", resync)
		return
	}
	plaintextCopy := make([]byte, len(plaintext))
	ocfbdec.XORKeyStream(plaintextCopy, ciphertext)

	if !bytes.Equal(plaintextCopy, plaintext) {
		t.Errorf("got: %x, want: %x (resync: %t)", plaintextCopy, plaintext, resync)
	}
}

func TestOCFB(t *testing.T) {
	testOCFB(t, OCFBNoResync)
	testOCFB(t, OCFBResync)
}
