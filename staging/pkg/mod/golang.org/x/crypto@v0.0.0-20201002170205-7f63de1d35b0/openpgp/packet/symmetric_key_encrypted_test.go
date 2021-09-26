// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/hex"
	"io"
	"io/ioutil"
	"testing"
)

func TestSymmetricKeyEncrypted(t *testing.T) {
	buf := readerFromHex(symmetricallyEncryptedHex)
	packet, err := Read(buf)
	if err != nil {
		t.Errorf("failed to read SymmetricKeyEncrypted: %s", err)
		return
	}
	ske, ok := packet.(*SymmetricKeyEncrypted)
	if !ok {
		t.Error("didn't find SymmetricKeyEncrypted packet")
		return
	}
	key, cipherFunc, err := ske.Decrypt([]byte("password"))
	if err != nil {
		t.Error(err)
		return
	}

	packet, err = Read(buf)
	if err != nil {
		t.Errorf("failed to read SymmetricallyEncrypted: %s", err)
		return
	}
	se, ok := packet.(*SymmetricallyEncrypted)
	if !ok {
		t.Error("didn't find SymmetricallyEncrypted packet")
		return
	}
	r, err := se.Decrypt(cipherFunc, key)
	if err != nil {
		t.Error(err)
		return
	}

	contents, err := ioutil.ReadAll(r)
	if err != nil && err != io.EOF {
		t.Error(err)
		return
	}

	expectedContents, _ := hex.DecodeString(symmetricallyEncryptedContentsHex)
	if !bytes.Equal(expectedContents, contents) {
		t.Errorf("bad contents got:%x want:%x", contents, expectedContents)
	}
}

const symmetricallyEncryptedHex = "8c0d04030302371a0b38d884f02060c91cf97c9973b8e58e028e9501708ccfe618fb92afef7fa2d80ddadd93cf"
const symmetricallyEncryptedContentsHex = "cb1062004d14c4df636f6e74656e74732e0a"

func TestSerializeSymmetricKeyEncryptedCiphers(t *testing.T) {
	tests := [...]struct {
		cipherFunc CipherFunction
		name       string
	}{
		{Cipher3DES, "Cipher3DES"},
		{CipherCAST5, "CipherCAST5"},
		{CipherAES128, "CipherAES128"},
		{CipherAES192, "CipherAES192"},
		{CipherAES256, "CipherAES256"},
	}

	for _, test := range tests {
		var buf bytes.Buffer
		passphrase := []byte("testing")
		config := &Config{
			DefaultCipher: test.cipherFunc,
		}

		key, err := SerializeSymmetricKeyEncrypted(&buf, passphrase, config)
		if err != nil {
			t.Errorf("cipher(%s) failed to serialize: %s", test.name, err)
			continue
		}

		p, err := Read(&buf)
		if err != nil {
			t.Errorf("cipher(%s) failed to reparse: %s", test.name, err)
			continue
		}

		ske, ok := p.(*SymmetricKeyEncrypted)
		if !ok {
			t.Errorf("cipher(%s) parsed a different packet type: %#v", test.name, p)
			continue
		}

		if ske.CipherFunc != config.DefaultCipher {
			t.Errorf("cipher(%s) SKE cipher function is %d (expected %d)", test.name, ske.CipherFunc, config.DefaultCipher)
		}
		parsedKey, parsedCipherFunc, err := ske.Decrypt(passphrase)
		if err != nil {
			t.Errorf("cipher(%s) failed to decrypt reparsed SKE: %s", test.name, err)
			continue
		}
		if !bytes.Equal(key, parsedKey) {
			t.Errorf("cipher(%s) keys don't match after Decrypt: %x (original) vs %x (parsed)", test.name, key, parsedKey)
		}
		if parsedCipherFunc != test.cipherFunc {
			t.Errorf("cipher(%s) cipher function doesn't match after Decrypt: %d (original) vs %d (parsed)",
				test.name, test.cipherFunc, parsedCipherFunc)
		}
	}
}
