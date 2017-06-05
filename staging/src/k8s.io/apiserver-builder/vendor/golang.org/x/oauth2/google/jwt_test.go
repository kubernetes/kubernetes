// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"strings"
	"testing"
	"time"

	"golang.org/x/oauth2/jws"
)

func TestJWTAccessTokenSourceFromJSON(t *testing.T) {
	// Generate a key we can use in the test data.
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	// Encode the key and substitute into our example JSON.
	enc := pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	})
	enc, err = json.Marshal(string(enc))
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	jsonKey := bytes.Replace(jwtJSONKey, []byte(`"super secret key"`), enc, 1)

	ts, err := JWTAccessTokenSourceFromJSON(jsonKey, "audience")
	if err != nil {
		t.Fatalf("JWTAccessTokenSourceFromJSON: %v\nJSON: %s", err, string(jsonKey))
	}

	tok, err := ts.Token()
	if err != nil {
		t.Fatalf("Token: %v", err)
	}

	if got, want := tok.TokenType, "Bearer"; got != want {
		t.Errorf("TokenType = %q, want %q", got, want)
	}
	if got := tok.Expiry; tok.Expiry.Before(time.Now()) {
		t.Errorf("Expiry = %v, should not be expired", got)
	}

	err = jws.Verify(tok.AccessToken, &privateKey.PublicKey)
	if err != nil {
		t.Errorf("jws.Verify on AccessToken: %v", err)
	}

	claim, err := jws.Decode(tok.AccessToken)
	if err != nil {
		t.Fatalf("jws.Decode on AccessToken: %v", err)
	}

	if got, want := claim.Iss, "gopher@developer.gserviceaccount.com"; got != want {
		t.Errorf("Iss = %q, want %q", got, want)
	}
	if got, want := claim.Sub, "gopher@developer.gserviceaccount.com"; got != want {
		t.Errorf("Sub = %q, want %q", got, want)
	}
	if got, want := claim.Aud, "audience"; got != want {
		t.Errorf("Aud = %q, want %q", got, want)
	}

	// Finally, check the header private key.
	parts := strings.Split(tok.AccessToken, ".")
	hdrJSON, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		t.Fatalf("base64 DecodeString: %v\nString: %q", err, parts[0])
	}
	var hdr jws.Header
	if err := json.Unmarshal([]byte(hdrJSON), &hdr); err != nil {
		t.Fatalf("json.Unmarshal: %v (%q)", err, hdrJSON)
	}

	if got, want := hdr.KeyID, "268f54e43a1af97cfc71731688434f45aca15c8b"; got != want {
		t.Errorf("Header KeyID = %q, want %q", got, want)
	}
}
