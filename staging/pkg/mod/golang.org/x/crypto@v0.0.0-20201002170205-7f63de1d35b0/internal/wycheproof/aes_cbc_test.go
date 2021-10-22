// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/hex"
	"fmt"
	"testing"
)

func TestAesCbc(t *testing.T) {
	// IndCpaTestVector
	type IndCpaTestVector struct {

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// the raw ciphertext (without IV)
		Ct string `json:"ct,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

		// the initialization vector
		Iv string `json:"iv,omitempty"`

		// the key
		Key string `json:"key,omitempty"`

		// the plaintext
		Msg string `json:"msg,omitempty"`

		// Test result
		Result string `json:"result,omitempty"`

		// Identifier of the test case
		TcId int `json:"tcId,omitempty"`
	}

	// Notes a description of the labels used in the test vectors
	type Notes struct {
	}

	// IndCpaTestGroup
	type IndCpaTestGroup struct {

		// the IV size in bits
		IvSize int `json:"ivSize,omitempty"`

		// the keySize in bits
		KeySize int `json:"keySize,omitempty"`

		// the expected size of the tag in bits
		TagSize int                 `json:"tagSize,omitempty"`
		Tests   []*IndCpaTestVector `json:"tests,omitempty"`
		Type    interface{}         `json:"type,omitempty"`
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
		NumberOfTests int                `json:"numberOfTests,omitempty"`
		Schema        interface{}        `json:"schema,omitempty"`
		TestGroups    []*IndCpaTestGroup `json:"testGroups,omitempty"`
	}

	var root Root
	readTestVector(t, "aes_cbc_pkcs5_test.json", &root)
	for _, tg := range root.TestGroups {
	tests:
		for _, tv := range tg.Tests {
			block, err := aes.NewCipher(decodeHex(tv.Key))
			if err != nil {
				t.Fatalf("#%d: %v", tv.TcId, err)
			}
			mode := cipher.NewCBCDecrypter(block, decodeHex(tv.Iv))
			ct := decodeHex(tv.Ct)
			if len(ct)%aes.BlockSize != 0 {
				panic(fmt.Sprintf("#%d: ciphertext is not a multiple of the block size", tv.TcId))
			}
			mode.CryptBlocks(ct, ct) // decrypt the block in place

			// Skip the tests that are broken due to bad padding. Panic if there are any
			// tests left that are invalid for some other reason in the future, to
			// evaluate what to do with those tests.
			for _, flag := range tv.Flags {
				if flag == "BadPadding" {
					continue tests
				}
			}
			if !shouldPass(tv.Result, tv.Flags, nil) {
				panic(fmt.Sprintf("#%d: found an invalid test that is broken for some reason other than bad padding", tv.TcId))
			}

			// Remove the PKCS#5 padding from the given ciphertext to validate it
			padding := ct[len(ct)-1]
			paddingNum := int(padding)
			for i := paddingNum; i > 0; i-- {
				if ct[len(ct)-i] != padding { // panic if the padding is unexpectedly bad
					panic(fmt.Sprintf("#%d: bad padding at index=%d of %v", tv.TcId, i, ct))
				}
			}
			ct = ct[:len(ct)-paddingNum]

			if got, want := hex.EncodeToString(ct), tv.Msg; got != want {
				t.Errorf("#%d, type: %s, comment: %q, decoded ciphertext not equal: %s, want %s", tv.TcId, tv.Result, tv.Comment, got, want)
			}
		}
	}
}
