// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jws

import (
	"crypto/rand"
	"crypto/rsa"
	"testing"
)

func TestSignAndVerify(t *testing.T) {
	header := &Header{
		Algorithm: "RS256",
		Typ:       "JWT",
	}
	payload := &ClaimSet{
		Iss: "http://google.com/",
		Aud: "",
		Exp: 3610,
		Iat: 10,
	}

	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	token, err := Encode(header, payload, privateKey)
	if err != nil {
		t.Fatal(err)
	}

	err = Verify(token, &privateKey.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
}

func TestVerifyFailsOnMalformedClaim(t *testing.T) {
	err := Verify("abc.def", nil)
	if err == nil {
		t.Error("got no errors; want improperly formed JWT not to be verified")
	}
}
