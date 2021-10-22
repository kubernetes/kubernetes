// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"bytes"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"testing"
)

func TestRSAOAEPDecrypt(t *testing.T) {
	// Notes a description of the labels used in the test vectors
	type Notes struct {
	}

	// RsaesOaepTestVector
	type RsaesOaepTestVector struct {

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// An encryption of msg
		Ct string `json:"ct,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

		// The label used for the encryption
		Label string `json:"label,omitempty"`

		// The encrypted message
		Msg string `json:"msg,omitempty"`

		// Test result
		Result string `json:"result,omitempty"`

		// Identifier of the test case
		TcId int `json:"tcId,omitempty"`
	}

	// RsaesOaepTestGroup
	type RsaesOaepTestGroup struct {

		// The private exponent
		D string `json:"d,omitempty"`

		// The public exponent
		E string `json:"e,omitempty"`

		// the message generating function (e.g. MGF1)
		Mgf string `json:"mgf,omitempty"`

		// The hash function used for the message generating function.
		MgfSha string `json:"mgfSha,omitempty"`

		// The modulus of the key
		N string `json:"n,omitempty"`

		// Pem encoded private key
		PrivateKeyPem string `json:"privateKeyPem,omitempty"`

		// Pkcs 8 encoded private key.
		PrivateKeyPkcs8 string `json:"privateKeyPkcs8,omitempty"`

		// The hash function for hashing the label.
		Sha   string                 `json:"sha,omitempty"`
		Tests []*RsaesOaepTestVector `json:"tests,omitempty"`
		Type  interface{}            `json:"type,omitempty"`
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
		NumberOfTests int                   `json:"numberOfTests,omitempty"`
		Schema        interface{}           `json:"schema,omitempty"`
		TestGroups    []*RsaesOaepTestGroup `json:"testGroups,omitempty"`
	}

	// rsa.DecryptOAEP doesn't support using a different hash for the
	// MGF and the label, so skip all of the test vectors that use
	// these unbalanced constructions. rsa_oaep_misc_test.json contains
	// both balanced and unbalanced constructions so in that case
	// we just filter out any test groups where MgfSha != Sha
	files := []string{
		"rsa_oaep_2048_sha1_mgf1sha1_test.json",
		"rsa_oaep_2048_sha224_mgf1sha224_test.json",
		"rsa_oaep_2048_sha256_mgf1sha256_test.json",
		"rsa_oaep_2048_sha384_mgf1sha384_test.json",
		"rsa_oaep_2048_sha512_mgf1sha512_test.json",
		"rsa_oaep_3072_sha256_mgf1sha256_test.json",
		"rsa_oaep_3072_sha512_mgf1sha512_test.json",
		"rsa_oaep_4096_sha256_mgf1sha256_test.json",
		"rsa_oaep_4096_sha512_mgf1sha512_test.json",
		"rsa_oaep_misc_test.json",
	}

	flagsShouldPass := map[string]bool{
		// rsa.DecryptOAEP happily supports small key sizes
		"SmallModulus": true,
	}

	for _, f := range files {
		var root Root
		readTestVector(t, f, &root)
		for _, tg := range root.TestGroups {
			if tg.MgfSha != tg.Sha {
				continue
			}
			priv, err := x509.ParsePKCS8PrivateKey(decodeHex(tg.PrivateKeyPkcs8))
			if err != nil {
				t.Fatalf("%s failed to parse PKCS #8 private key: %s", f, err)
			}
			hash := parseHash(tg.Sha)
			for _, tv := range tg.Tests {
				t.Run(fmt.Sprintf("%s #%d", f, tv.TcId), func(t *testing.T) {
					wantPass := shouldPass(tv.Result, tv.Flags, flagsShouldPass)
					plaintext, err := rsa.DecryptOAEP(hash.New(), nil, priv.(*rsa.PrivateKey), decodeHex(tv.Ct), decodeHex(tv.Label))
					if wantPass {
						if err != nil {
							t.Fatalf("comment: %s, expected success: %s", tv.Comment, err)
						}
						if !bytes.Equal(plaintext, decodeHex(tv.Msg)) {
							t.Errorf("comment: %s, unexpected plaintext: got %x, want %s", tv.Comment, plaintext, tv.Msg)
						}
					} else if err == nil {
						t.Errorf("comment: %s, expected failure", tv.Comment)
					}
				})
			}
		}
	}
}
