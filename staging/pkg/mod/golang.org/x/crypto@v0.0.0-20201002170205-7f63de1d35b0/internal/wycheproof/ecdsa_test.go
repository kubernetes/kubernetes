// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wycheproof

import (
	"crypto/ecdsa"
	"testing"
)

func TestEcdsa(t *testing.T) {
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

	// EcPublicKey
	type EcPublicKey struct {

		// the EC group used by this public key
		Curve interface{} `json:"curve,omitempty"`

		// the key size in bits
		KeySize int `json:"keySize,omitempty"`

		// the key type
		Type string `json:"type,omitempty"`

		// encoded public key point
		Uncompressed string `json:"uncompressed,omitempty"`

		// the x-coordinate of the public key point
		Wx string `json:"wx,omitempty"`

		// the y-coordinate of the public key point
		Wy string `json:"wy,omitempty"`
	}

	// EcUnnamedGroup
	type EcUnnamedGroup struct {

		// coefficient a of the elliptic curve equation
		A string `json:"a,omitempty"`

		// coefficient b of the elliptic curve equation
		B string `json:"b,omitempty"`

		// the x-coordinate of the generator
		Gx string `json:"gx,omitempty"`

		// the y-coordinate of the generator
		Gy string `json:"gy,omitempty"`

		// the cofactor
		H int `json:"h,omitempty"`

		// the order of the generator
		N string `json:"n,omitempty"`

		// the order of the underlying field
		P string `json:"p,omitempty"`

		// an unnamed EC group over a prime field in Weierstrass form
		Type string `json:"type,omitempty"`
	}

	// EcdsaTestGroup
	type EcdsaTestGroup struct {

		// unenocded EC public key
		Key *EcPublicKey `json:"key,omitempty"`

		// DER encoded public key
		KeyDer string `json:"keyDer,omitempty"`

		// Pem encoded public key
		KeyPem string `json:"keyPem,omitempty"`

		// the hash function used for ECDSA
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
		NumberOfTests int               `json:"numberOfTests,omitempty"`
		Schema        interface{}       `json:"schema,omitempty"`
		TestGroups    []*EcdsaTestGroup `json:"testGroups,omitempty"`
	}

	flagsShouldPass := map[string]bool{
		// An encoded ASN.1 integer missing a leading zero is invalid, but accepted by some implementations.
		"MissingZero": false,
		// A signature using a weaker hash than the EC params is not a security risk, as long as the hash is secure.
		// https://www.imperialviolet.org/2014/05/25/strengthmatching.html
		"WeakHash": true,
	}

	// supportedCurves is a map of all elliptic curves supported
	// by crypto/elliptic, which can subsequently be parsed and tested.
	supportedCurves := map[string]bool{
		"secp224r1": true,
		"secp256r1": true,
		"secp384r1": true,
		"secp521r1": true,
	}

	var root Root
	readTestVector(t, "ecdsa_test.json", &root)
	for _, tg := range root.TestGroups {
		curve := tg.Key.Curve.(string)
		if !supportedCurves[curve] {
			continue
		}
		pub := decodePublicKey(tg.KeyDer).(*ecdsa.PublicKey)
		h := parseHash(tg.Sha).New()
		for _, sig := range tg.Tests {
			h.Reset()
			h.Write(decodeHex(sig.Msg))
			hashed := h.Sum(nil)
			got := verifyASN1(pub, hashed, decodeHex(sig.Sig))
			if want := shouldPass(sig.Result, sig.Flags, flagsShouldPass); got != want {
				t.Errorf("tcid: %d, type: %s, comment: %q, wanted success: %t", sig.TcId, sig.Result, sig.Comment, want)
			}
		}
	}
}
