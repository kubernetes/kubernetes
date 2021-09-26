// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/rsa"
	"testing"
)

func TestRsa(t *testing.T) {
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
		// Omitting the parameter field in an ASN encoded integer is a legacy behavior.
		"MissingNull": false,
		// Keys with a modulus less than 2048 bits are supported by crypto/rsa.
		"SmallModulus": true,
		// Small public keys are supported by crypto/rsa.
		"SmallPublicKey": true,
	}

	var root Root
	readTestVector(t, "rsa_signature_test.json", &root)
	for _, tg := range root.TestGroups {
		pub := decodePublicKey(tg.KeyDer).(*rsa.PublicKey)
		ch := parseHash(tg.Sha)
		h := ch.New()
		for _, sig := range tg.Tests {
			h.Reset()
			h.Write(decodeHex(sig.Msg))
			hashed := h.Sum(nil)
			err := rsa.VerifyPKCS1v15(pub, ch, hashed, decodeHex(sig.Sig))
			want := shouldPass(sig.Result, sig.Flags, flagsShouldPass)
			if (err == nil) != want {
				t.Errorf("tcid: %d, type: %s, comment: %q, wanted success: %t", sig.TcId, sig.Result, sig.Comment, want)
			}
		}
	}
}
