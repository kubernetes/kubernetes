// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.13

package wycheproof

import (
	"testing"

	"golang.org/x/crypto/ed25519"
)

func TestEddsa(t *testing.T) {
	// Jwk the private key in webcrypto format
	type Jwk struct {
	}

	// Key unencoded key pair
	type Key struct {
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

	// EddsaTestGroup
	type EddsaTestGroup struct {

		// the private key in webcrypto format
		Jwk *Jwk `json:"jwk,omitempty"`

		// unencoded key pair
		Key *Key `json:"key,omitempty"`

		// Asn encoded public key
		KeyDer string `json:"keyDer,omitempty"`

		// Pem encoded public key
		KeyPem string                 `json:"keyPem,omitempty"`
		Tests  []*SignatureTestVector `json:"tests,omitempty"`
		Type   interface{}            `json:"type,omitempty"`
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
		NumberOfTests int               `json:"numberOfTests,omitempty"`
		Schema        interface{}       `json:"schema,omitempty"`
		TestGroups    []*EddsaTestGroup `json:"testGroups,omitempty"`
	}

	var root Root
	readTestVector(t, "eddsa_test.json", &root)
	for _, tg := range root.TestGroups {
		pub := decodePublicKey(tg.KeyDer).(ed25519.PublicKey)
		for _, sig := range tg.Tests {
			got := ed25519.Verify(pub, decodeHex(sig.Msg), decodeHex(sig.Sig))
			if want := shouldPass(sig.Result, sig.Flags, nil); got != want {
				t.Errorf("tcid: %d, type: %s, comment: %q, wanted success: %t", sig.TcId, sig.Result, sig.Comment, want)
			}
		}
	}
}
