// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/hmac"
	"testing"
)

func TestHMAC(t *testing.T) {
	// MacTestVector
	type MacTestVector struct {

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

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

	// MacTestGroup
	type MacTestGroup struct {

		// the keySize in bits
		KeySize int `json:"keySize,omitempty"`

		// the expected size of the tag in bits
		TagSize int              `json:"tagSize,omitempty"`
		Tests   []*MacTestVector `json:"tests,omitempty"`
		Type    interface{}      `json:"type,omitempty"`
	}

	// Notes a description of the labels used in the test vectors
	type Notes struct {
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
		NumberOfTests int             `json:"numberOfTests,omitempty"`
		Schema        interface{}     `json:"schema,omitempty"`
		TestGroups    []*MacTestGroup `json:"testGroups,omitempty"`
	}

	fileHashAlgs := map[string]string{
		"hmac_sha1_test.json":   "SHA-1",
		"hmac_sha224_test.json": "SHA-224",
		"hmac_sha256_test.json": "SHA-256",
		"hmac_sha384_test.json": "SHA-384",
		"hmac_sha512_test.json": "SHA-512",
	}

	for f := range fileHashAlgs {
		var root Root
		readTestVector(t, f, &root)
		for _, tg := range root.TestGroups {
			h := parseHash(fileHashAlgs[f])
			// Skip test vectors where the tag length does not equal the
			// hash length, since crypto/hmac does not support generating
			// these truncated tags.
			if tg.TagSize/8 != h.Size() {
				continue
			}
			for _, tv := range tg.Tests {
				hm := hmac.New(h.New, decodeHex(tv.Key))
				hm.Write(decodeHex(tv.Msg))
				tag := hm.Sum(nil)
				got := hmac.Equal(decodeHex(tv.Tag), tag)
				if want := shouldPass(tv.Result, tv.Flags, nil); want != got {
					t.Errorf("%s, tcid: %d, type: %s, comment: %q, unexpected result", f, tv.TcId, tv.Result, tv.Comment)
				}
			}
		}
	}
}
