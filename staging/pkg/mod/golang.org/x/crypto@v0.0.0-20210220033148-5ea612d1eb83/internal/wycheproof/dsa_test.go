// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/dsa"
	"testing"

	wdsa "golang.org/x/crypto/internal/wycheproof/internal/dsa"
)

func TestDsa(t *testing.T) {
	// AsnSignatureTestVector
	type AsnSignatureTestVector struct {

		// A brief description of the test case
		Comment string `json:"comment,omitempty"`

		// A list of flags
		Flags []string `json:"flags,omitempty"`

		// The message to sign
		Msg string `json:"msg,omitempty"`

		// Test result
		Result string `json:"result,omitempty"`

		// An ASN encoded signature for msg
		Sig string `json:"sig,omitempty"`

		// Identifier of the test case
		TcId int `json:"tcId,omitempty"`
	}

	// DsaPublicKey
	type DsaPublicKey struct {

		// the generator of the multiplicative subgroup
		G string `json:"g,omitempty"`

		// the key size in bits
		KeySize int `json:"keySize,omitempty"`

		// the modulus p
		P string `json:"p,omitempty"`

		// the order of the generator g
		Q string `json:"q,omitempty"`

		// the key type
		Type string `json:"type,omitempty"`

		// the public key value
		Y string `json:"y,omitempty"`
	}

	// DsaTestGroup
	type DsaTestGroup struct {

		// unenocded DSA public key
		Key *DsaPublicKey `json:"key,omitempty"`

		// DER encoded public key
		KeyDer string `json:"keyDer,omitempty"`

		// Pem encoded public key
		KeyPem string `json:"keyPem,omitempty"`

		// the hash function used for DSA
		Sha   string                    `json:"sha,omitempty"`
		Tests []*AsnSignatureTestVector `json:"tests,omitempty"`
		Type  interface{}               `json:"type,omitempty"`
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
		TestGroups    []*DsaTestGroup `json:"testGroups,omitempty"`
	}

	flagsShouldPass := map[string]bool{
		// An encoded ASN.1 integer missing a leading zero is invalid, but accepted by some implementations.
		"NoLeadingZero": false,
	}

	var root Root
	readTestVector(t, "dsa_test.json", &root)
	for _, tg := range root.TestGroups {
		pub := decodePublicKey(tg.KeyDer).(*dsa.PublicKey)
		h := parseHash(tg.Sha).New()
		for _, sig := range tg.Tests {
			h.Reset()
			h.Write(decodeHex(sig.Msg))
			hashed := h.Sum(nil)
			hashed = hashed[:pub.Q.BitLen()/8] // Truncate to the byte-length of the subgroup (Q)
			got := wdsa.VerifyASN1(pub, hashed, decodeHex(sig.Sig))
			if want := shouldPass(sig.Result, sig.Flags, flagsShouldPass); got != want {
				t.Errorf("tcid: %d, type: %s, comment: %q, wanted success: %t", sig.TcId, sig.Result, sig.Comment, want)
			}
		}
	}
}
