// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/rsa"
	"testing"
)

func TestRsaPss(t *testing.T) {
	// KeyJwk Public key in JWK format
	type KeyJwk struct {
	}

	// Notes a description of the labels used in the test vectors
	type Notes struct {
	}

	// SignatureTestVector
	type SignatureTestVector struct {

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

		// The message to sign
		Msg string `json:"msg,omitempty"`

		// Test result
		Result string `json:"result,omitempty"`

		// A signature for msg
		Sig string `json:"sig,omitempty"`

		// Identifier of the test case
		TcId int `json:"tcId,omitempty"`
	}

	// RsassaPkcs1TestGroup
	type RsassaPkcs1TestGroup struct {

		// The private exponent
		D string `json:"d,omitempty"`

		// The public exponent
		E string `json:"e,omitempty"`

		// ASN encoding of the sequence [n, e]
		KeyAsn string `json:"keyAsn,omitempty"`

		// ASN encoding of the public key
		KeyDer string `json:"keyDer,omitempty"`

		// Public key in JWK format
		KeyJwk *KeyJwk `json:"keyJwk,omitempty"`

		// Pem encoded public key
		KeyPem string `json:"keyPem,omitempty"`

		// the size of the modulus in bits
		KeySize int `json:"keySize,omitempty"`

		// The modulus of the key
		N string `json:"n,omitempty"`

		// The salt length
		SLen int `json:"sLen,omitempty"`

		// the hash function used for the message
		Sha   string                 `json:"sha,omitempty"`
		Tests []*SignatureTestVector `json:"tests,omitempty"`
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
		NumberOfTests int                     `json:"numberOfTests,omitempty"`
		Schema        interface{}             `json:"schema,omitempty"`
		TestGroups    []*RsassaPkcs1TestGroup `json:"testGroups,omitempty"`
	}

	flagsShouldPass := map[string]bool{
		// A signature using a weaker hash than the EC params is not a security risk, as long as the hash is secure.
		// https://www.imperialviolet.org/2014/05/25/strengthmatching.html
		"WeakHash": true,
	}

	// filesOverrideToPassZeroSLen is a map of all test files
	// and which TcIds that should be overriden to pass if the
	// rsa.PSSOptions.SaltLength is zero.
	// These tests expect a failure with a PSSOptions.SaltLength: 0
	// and a signature that uses a different salt length. However,
	// a salt length of 0 is defined as rsa.PSSSaltLengthAuto which
	// works deterministically to auto-detect the length when
	// verifying, so these tests actually pass as they should.
	filesOverrideToPassZeroSLen := map[string][]int{
		"rsa_pss_2048_sha1_mgf1_20_test.json":       []int{46, 47},
		"rsa_pss_2048_sha256_mgf1_0_test.json":      []int{67, 68},
		"rsa_pss_2048_sha256_mgf1_32_test.json":     []int{67, 68},
		"rsa_pss_2048_sha512_256_mgf1_28_test.json": []int{13, 14, 15},
		"rsa_pss_2048_sha512_256_mgf1_32_test.json": []int{13, 14},
		"rsa_pss_3072_sha256_mgf1_32_test.json":     []int{67, 68},
		"rsa_pss_4096_sha256_mgf1_32_test.json":     []int{67, 68},
		"rsa_pss_4096_sha512_mgf1_32_test.json":     []int{136, 137},
		// "rsa_pss_misc_test.json": nil,  // TODO: This ones seems to be broken right now, but can enable later on.
	}

	for f := range filesOverrideToPassZeroSLen {
		var root Root
		readTestVector(t, f, &root)
		for _, tg := range root.TestGroups {
			pub := decodePublicKey(tg.KeyDer).(*rsa.PublicKey)
			ch := parseHash(tg.Sha)
			h := ch.New()
			opts := &rsa.PSSOptions{
				Hash:       ch,
				SaltLength: rsa.PSSSaltLengthAuto,
			}
			// Run all the tests twice: the first time with the salt length
			// as PSSSaltLengthAuto, and the second time with the salt length
			// explictily set to tg.SLen.
			for i := 0; i < 2; i++ {
				for _, sig := range tg.Tests {
					h.Reset()
					h.Write(decodeHex(sig.Msg))
					hashed := h.Sum(nil)
					err := rsa.VerifyPSS(pub, ch, hashed, decodeHex(sig.Sig), opts)
					want := shouldPass(sig.Result, sig.Flags, flagsShouldPass)
					if opts.SaltLength == 0 {
						for _, id := range filesOverrideToPassZeroSLen[f] {
							if sig.TcId == id {
								want = true
								break
							}
						}
					}
					if (err == nil) != want {
						t.Errorf("file: %v, tcid: %d, type: %s, opts.SaltLength: %v, comment: %q, wanted success: %t", f, sig.TcId, sig.Result, opts.SaltLength, sig.Comment, want)
					}
				}
				// Update opts.SaltLength for the second run of the tests.
				opts.SaltLength = tg.SLen
			}
		}
	}
}
