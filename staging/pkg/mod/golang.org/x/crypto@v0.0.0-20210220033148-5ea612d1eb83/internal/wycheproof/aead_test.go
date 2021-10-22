// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"fmt"
	"testing"

	"golang.org/x/crypto/chacha20poly1305"
)

func TestAEAD(t *testing.T) {
	// AeadTestVector
	type AeadTestVector struct {

		// additional authenticated data
		Aad string `json:"aad,omitempty"`

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// the ciphertext (without iv and tag)
		Ct string `json:"ct,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

		// the nonce
		Iv string `json:"iv,omitempty"`

		// the key
		Key string `json:"key,omitempty"`

		// the plaintext
		Msg string `json:"msg,omitempty"`

		// Test result
		Result string `json:"result,omitempty"`

		// the authentication tag
		Tag string `json:"tag,omitempty"`

		// Identifier of the test case
		TcId int `json:"tcId,omitempty"`
	}

	// Notes a description of the labels used in the test vectors
	type Notes struct {
	}

	// AeadTestGroup
	type AeadTestGroup struct {

		// the IV size in bits
		IvSize int `json:"ivSize,omitempty"`

		// the keySize in bits
		KeySize int `json:"keySize,omitempty"`

		// the expected size of the tag in bits
		TagSize int               `json:"tagSize,omitempty"`
		Tests   []*AeadTestVector `json:"tests,omitempty"`
		Type    interface{}       `json:"type,omitempty"`
	}

	// Root
	type Root struct {

		// the primitive tested in the test file
		Algorithm string `json:"algorithm,omitempty"`

		// the version of the test vectors.
		GeneratorVersion string `json:"generatorVersion,omitempty"`

		// additional documentation
		Header []string `json:"header,omitempty"`

		// a description of the labels used in the test vectors
		Notes *Notes `json:"notes,omitempty"`

		// the number of test vectors in this test
		NumberOfTests int              `json:"numberOfTests,omitempty"`
		Schema        interface{}      `json:"schema,omitempty"`
		TestGroups    []*AeadTestGroup `json:"testGroups,omitempty"`
	}

	testSealOpen := func(t *testing.T, aead cipher.AEAD, tv *AeadTestVector, recoverBadNonce func()) {
		defer recoverBadNonce()

		iv, tag, ct, msg, aad := decodeHex(tv.Iv), decodeHex(tv.Tag), decodeHex(tv.Ct), decodeHex(tv.Msg), decodeHex(tv.Aad)

		genCT := aead.Seal(nil, iv, msg, aad)
		genMsg, err := aead.Open(nil, iv, genCT, aad)
		if err != nil {
			t.Errorf("failed to decrypt generated ciphertext: %s", err)
		}
		if !bytes.Equal(genMsg, msg) {
			t.Errorf("unexpected roundtripped plaintext: got %x, want %x", genMsg, msg)
		}

		ctWithTag := append(ct, tag...)
		msg2, err := aead.Open(nil, iv, ctWithTag, aad)
		wantPass := shouldPass(tv.Result, tv.Flags, nil)
		if !wantPass && err == nil {
			t.Error("decryption succeeded when it should've failed")
		} else if wantPass {
			if err != nil {
				t.Fatalf("decryption failed: %s", err)
			}
			if !bytes.Equal(genCT, ctWithTag) {
				t.Errorf("generated ciphertext doesn't match expected: got %x, want %x", genCT, ctWithTag)
			}
			if !bytes.Equal(msg, msg2) {
				t.Errorf("decrypted ciphertext doesn't match expected: got %x, want %x", msg2, msg)
			}
		}
	}

	vectors := map[string]func(*testing.T, []byte) cipher.AEAD{
		"aes_gcm_test.json": func(t *testing.T, key []byte) cipher.AEAD {
			aesCipher, err := aes.NewCipher(key)
			if err != nil {
				t.Fatalf("failed to construct cipher: %s", err)
			}
			aead, err := cipher.NewGCM(aesCipher)
			if err != nil {
				t.Fatalf("failed to construct cipher: %s", err)
			}
			return aead
		},
		"chacha20_poly1305_test.json": func(t *testing.T, key []byte) cipher.AEAD {
			aead, err := chacha20poly1305.New(key)
			if err != nil {
				t.Fatalf("failed to construct cipher: %s", err)
			}
			return aead
		},
		"xchacha20_poly1305_test.json": func(t *testing.T, key []byte) cipher.AEAD {
			aead, err := chacha20poly1305.NewX(key)
			if err != nil {
				t.Fatalf("failed to construct cipher: %s", err)
			}
			return aead
		},
	}
	for file, cipherInit := range vectors {
		var root Root
		readTestVector(t, file, &root)
		for _, tg := range root.TestGroups {
			for _, tv := range tg.Tests {
				testName := fmt.Sprintf("%s #%d", file, tv.TcId)
				if tv.Comment != "" {
					testName += fmt.Sprintf(" %s", tv.Comment)
				}
				t.Run(testName, func(t *testing.T) {
					aead := cipherInit(t, decodeHex(tv.Key))
					testSealOpen(t, aead, tv, func() {
						// A bad nonce causes a panic in AEAD.Seal and AEAD.Open,
						// so should be recovered. Fail the test if it broke for
						// some other reason.
						if r := recover(); r != nil {
							if tg.IvSize/8 == aead.NonceSize() {
								t.Error("unexpected panic")
							}
						}
					})
				})
			}
		}
	}
}
