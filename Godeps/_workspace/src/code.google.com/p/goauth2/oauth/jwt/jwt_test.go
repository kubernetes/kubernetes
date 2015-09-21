// Copyright 2012 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For package documentation please see jwt.go.
//
package jwt

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"io/ioutil"
	"net/http"
	"testing"
	"time"
)

const (
	stdHeaderStr = `{"alg":"RS256","typ":"JWT"}`
	iss          = "761326798069-r5mljlln1rd4lrbhg75efgigp36m78j5@developer.gserviceaccount.com"
	scope        = "https://www.googleapis.com/auth/prediction"
	exp          = 1328554385
	iat          = 1328550785 // exp + 1 hour
)

// Base64url encoded Header
const headerEnc = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"

// Base64url encoded ClaimSet
const claimSetEnc = "eyJpc3MiOiI3NjEzMjY3OTgwNjktcjVtbGpsbG4xcmQ0bHJiaGc3NWVmZ2lncDM2bTc4ajVAZGV2ZWxvcGVyLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJzY29wZSI6Imh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL2F1dGgvcHJlZGljdGlvbiIsImF1ZCI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi90b2tlbiIsImV4cCI6MTMyODU1NDM4NSwiaWF0IjoxMzI4NTUwNzg1fQ"

// Base64url encoded Signature
const sigEnc = "olukbHreNiYrgiGCTEmY3eWGeTvYDSUHYoE84Jz3BRPBSaMdZMNOn_0CYK7UHPO7OdvUofjwft1dH59UxE9GWS02pjFti1uAQoImaqjLZoTXr8qiF6O_kDa9JNoykklWlRAIwGIZkDupCS-8cTAnM_ksSymiH1coKJrLDUX_BM0x2f4iMFQzhL5vT1ll-ZipJ0lNlxb5QsyXxDYcxtHYguF12-vpv3ItgT0STfcXoWzIGQoEbhwB9SBp9JYcQ8Ygz6pYDjm0rWX9LrchmTyDArCodpKLFtutNgcIFUP9fWxvwd1C2dNw5GjLcKr9a_SAERyoJ2WnCR1_j9N0wD2o0g"

// Base64url encoded Token
const tokEnc = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI3NjEzMjY3OTgwNjktcjVtbGpsbG4xcmQ0bHJiaGc3NWVmZ2lncDM2bTc4ajVAZGV2ZWxvcGVyLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJzY29wZSI6Imh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL2F1dGgvcHJlZGljdGlvbiIsImF1ZCI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi90b2tlbiIsImV4cCI6MTMyODU1NDM4NSwiaWF0IjoxMzI4NTUwNzg1fQ.olukbHreNiYrgiGCTEmY3eWGeTvYDSUHYoE84Jz3BRPBSaMdZMNOn_0CYK7UHPO7OdvUofjwft1dH59UxE9GWS02pjFti1uAQoImaqjLZoTXr8qiF6O_kDa9JNoykklWlRAIwGIZkDupCS-8cTAnM_ksSymiH1coKJrLDUX_BM0x2f4iMFQzhL5vT1ll-ZipJ0lNlxb5QsyXxDYcxtHYguF12-vpv3ItgT0STfcXoWzIGQoEbhwB9SBp9JYcQ8Ygz6pYDjm0rWX9LrchmTyDArCodpKLFtutNgcIFUP9fWxvwd1C2dNw5GjLcKr9a_SAERyoJ2WnCR1_j9N0wD2o0g"

// Private key for testing
const privateKeyPem = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA4ej0p7bQ7L/r4rVGUz9RN4VQWoej1Bg1mYWIDYslvKrk1gpj
7wZgkdmM7oVK2OfgrSj/FCTkInKPqaCR0gD7K80q+mLBrN3PUkDrJQZpvRZIff3/
xmVU1WeruQLFJjnFb2dqu0s/FY/2kWiJtBCakXvXEOb7zfbINuayL+MSsCGSdVYs
SliS5qQpgyDap+8b5fpXZVJkq92hrcNtbkg7hCYUJczt8n9hcCTJCfUpApvaFQ18
pe+zpyl4+WzkP66I28hniMQyUlA1hBiskT7qiouq0m8IOodhv2fagSZKjOTTU2xk
SBc//fy3ZpsL7WqgsZS7Q+0VRK8gKfqkxg5OYQIDAQABAoIBAQDGGHzQxGKX+ANk
nQi53v/c6632dJKYXVJC+PDAz4+bzU800Y+n/bOYsWf/kCp94XcG4Lgsdd0Gx+Zq
HD9CI1IcqqBRR2AFscsmmX6YzPLTuEKBGMW8twaYy3utlFxElMwoUEsrSWRcCA1y
nHSDzTt871c7nxCXHxuZ6Nm/XCL7Bg8uidRTSC1sQrQyKgTPhtQdYrPQ4WZ1A4J9
IisyDYmZodSNZe5P+LTJ6M1SCgH8KH9ZGIxv3diMwzNNpk3kxJc9yCnja4mjiGE2
YCNusSycU5IhZwVeCTlhQGcNeV/skfg64xkiJE34c2y2ttFbdwBTPixStGaF09nU
Z422D40BAoGBAPvVyRRsC3BF+qZdaSMFwI1yiXY7vQw5+JZh01tD28NuYdRFzjcJ
vzT2n8LFpj5ZfZFvSMLMVEFVMgQvWnN0O6xdXvGov6qlRUSGaH9u+TCPNnIldjMP
B8+xTwFMqI7uQr54wBB+Poq7dVRP+0oHb0NYAwUBXoEuvYo3c/nDoRcZAoGBAOWl
aLHjMv4CJbArzT8sPfic/8waSiLV9Ixs3Re5YREUTtnLq7LoymqB57UXJB3BNz/2
eCueuW71avlWlRtE/wXASj5jx6y5mIrlV4nZbVuyYff0QlcG+fgb6pcJQuO9DxMI
aqFGrWP3zye+LK87a6iR76dS9vRU+bHZpSVvGMKJAoGAFGt3TIKeQtJJyqeUWNSk
klORNdcOMymYMIlqG+JatXQD1rR6ThgqOt8sgRyJqFCVT++YFMOAqXOBBLnaObZZ
CFbh1fJ66BlSjoXff0W+SuOx5HuJJAa5+WtFHrPajwxeuRcNa8jwxUsB7n41wADu
UqWWSRedVBg4Ijbw3nWwYDECgYB0pLew4z4bVuvdt+HgnJA9n0EuYowVdadpTEJg
soBjNHV4msLzdNqbjrAqgz6M/n8Ztg8D2PNHMNDNJPVHjJwcR7duSTA6w2p/4k28
bvvk/45Ta3XmzlxZcZSOct3O31Cw0i2XDVc018IY5be8qendDYM08icNo7vQYkRH
504kQQKBgQDjx60zpz8ozvm1XAj0wVhi7GwXe+5lTxiLi9Fxq721WDxPMiHDW2XL
YXfFVy/9/GIMvEiGYdmarK1NW+VhWl1DC5xhDg0kvMfxplt4tynoq1uTsQTY31Mx
BeF5CT/JuNYk3bEBF0H/Q3VGO1/ggVS+YezdFbLWIRoMnLj6XCFEGg==
-----END RSA PRIVATE KEY-----`

// Public key to go with the private key for testing
const publicKeyPem = `-----BEGIN CERTIFICATE-----
MIIDIzCCAgugAwIBAgIJAMfISuBQ5m+5MA0GCSqGSIb3DQEBBQUAMBUxEzARBgNV
BAMTCnVuaXQtdGVzdHMwHhcNMTExMjA2MTYyNjAyWhcNMjExMjAzMTYyNjAyWjAV
MRMwEQYDVQQDEwp1bml0LXRlc3RzMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEA4ej0p7bQ7L/r4rVGUz9RN4VQWoej1Bg1mYWIDYslvKrk1gpj7wZgkdmM
7oVK2OfgrSj/FCTkInKPqaCR0gD7K80q+mLBrN3PUkDrJQZpvRZIff3/xmVU1Wer
uQLFJjnFb2dqu0s/FY/2kWiJtBCakXvXEOb7zfbINuayL+MSsCGSdVYsSliS5qQp
gyDap+8b5fpXZVJkq92hrcNtbkg7hCYUJczt8n9hcCTJCfUpApvaFQ18pe+zpyl4
+WzkP66I28hniMQyUlA1hBiskT7qiouq0m8IOodhv2fagSZKjOTTU2xkSBc//fy3
ZpsL7WqgsZS7Q+0VRK8gKfqkxg5OYQIDAQABo3YwdDAdBgNVHQ4EFgQU2RQ8yO+O
gN8oVW2SW7RLrfYd9jEwRQYDVR0jBD4wPIAU2RQ8yO+OgN8oVW2SW7RLrfYd9jGh
GaQXMBUxEzARBgNVBAMTCnVuaXQtdGVzdHOCCQDHyErgUOZvuTAMBgNVHRMEBTAD
AQH/MA0GCSqGSIb3DQEBBQUAA4IBAQBRv+M/6+FiVu7KXNjFI5pSN17OcW5QUtPr
odJMlWrJBtynn/TA1oJlYu3yV5clc/71Vr/AxuX5xGP+IXL32YDF9lTUJXG/uUGk
+JETpKmQviPbRsvzYhz4pf6ZIOZMc3/GIcNq92ECbseGO+yAgyWUVKMmZM0HqXC9
ovNslqe0M8C1sLm1zAR5z/h/litE7/8O2ietija3Q/qtl2TOXJdCA6sgjJX2WUql
ybrC55ct18NKf3qhpcEkGQvFU40rVYApJpi98DiZPYFdx1oBDp/f4uZ3ojpxRVFT
cDwcJLfNRCPUhormsY7fDS9xSyThiHsW9mjJYdcaKQkwYZ0F11yB
-----END CERTIFICATE-----`

var (
	privateKeyPemBytes = []byte(privateKeyPem)
	publicKeyPemBytes  = []byte(publicKeyPem)
	stdHeader          = &Header{Algorithm: stdAlgorithm, Type: stdType}
)

// Testing the urlEncode function.
func TestUrlEncode(t *testing.T) {
	enc := base64Encode([]byte(stdHeaderStr))
	b := []byte(enc)
	if b[len(b)-1] == 61 {
		t.Error("TestUrlEncode: last chat == \"=\"")
	}
	if enc != headerEnc {
		t.Error("TestUrlEncode: enc != headerEnc")
		t.Errorf("        enc = %s", enc)
		t.Errorf("  headerEnc = %s", headerEnc)
	}
}

// Test that the times are set properly.
func TestClaimSetSetTimes(t *testing.T) {
	c := &ClaimSet{
		Iss:   iss,
		Scope: scope,
	}
	iat := time.Unix(iat, 0)
	c.setTimes(iat)
	if c.exp.Unix() != exp {
		t.Error("TestClaimSetSetTimes: c.exp != exp")
		t.Errorf("  c.Exp = %d", c.exp.Unix())
		t.Errorf("    exp = %d", exp)
	}
}

// Given a well formed ClaimSet, test for proper encoding.
func TestClaimSetEncode(t *testing.T) {
	c := &ClaimSet{
		Iss:   iss,
		Scope: scope,
		exp:   time.Unix(exp, 0),
		iat:   time.Unix(iat, 0),
	}
	enc := c.encode()
	re, err := base64Decode(enc)
	if err != nil {
		t.Fatalf("error decoding encoded claim set: %v", err)
	}

	wa, err := base64Decode(claimSetEnc)
	if err != nil {
		t.Fatalf("error decoding encoded expected claim set: %v", err)
	}

	if enc != claimSetEnc {
		t.Error("TestClaimSetEncode: enc != claimSetEnc")
		t.Errorf("          enc = %s", string(re))
		t.Errorf("  claimSetEnc = %s", string(wa))
	}
}

// Test that claim sets with private claim names are encoded correctly.
func TestClaimSetWithPrivateNameEncode(t *testing.T) {
	iatT := time.Unix(iat, 0)
	expT := time.Unix(exp, 0)

	i, err := json.Marshal(iatT.Unix())
	if err != nil {
		t.Fatalf("error marshaling iatT value of %v: %v", iatT.Unix(), err)
	}
	iatStr := string(i)
	e, err := json.Marshal(expT.Unix())
	if err != nil {
		t.Fatalf("error marshaling expT value of %v: %v", expT.Unix(), err)
	}

	expStr := string(e)

	testCases := []struct {
		desc  string
		input map[string]interface{}
		want  string
	}{
		// Test a simple int field.
		{
			"single simple field",
			map[string]interface{}{"amount": 22},
			`{` +
				`"iss":"` + iss + `",` +
				`"scope":"` + scope + `",` +
				`"aud":"` + stdAud + `",` +
				`"exp":` + expStr + `,` +
				`"iat":` + iatStr + `,` +
				`"amount":22` +
				`}`,
		},
		{
			"multiple simple fields",
			map[string]interface{}{"tracking_code": "axZf", "amount": 22},
			`{` +
				`"iss":"` + iss + `",` +
				`"scope":"` + scope + `",` +
				`"aud":"` + stdAud + `",` +
				`"exp":` + expStr + `,` +
				`"iat":` + iatStr + `,` +
				`"amount":22,` +
				`"tracking_code":"axZf"` +
				`}`,
		},
		{
			"nested struct fields",
			map[string]interface{}{
				"tracking_code": "axZf",
				"purchase": struct {
					Description string `json:"desc"`
					Quantity    int32  `json:"q"`
					Time        int64  `json:"t"`
				}{
					"toaster",
					5,
					iat,
				},
			},
			`{` +
				`"iss":"` + iss + `",` +
				`"scope":"` + scope + `",` +
				`"aud":"` + stdAud + `",` +
				`"exp":` + expStr + `,` +
				`"iat":` + iatStr + `,` +
				`"purchase":{"desc":"toaster","q":5,"t":` + iatStr + `},` +
				`"tracking_code":"axZf"` +
				`}`,
		},
	}

	for _, testCase := range testCases {
		c := &ClaimSet{
			Iss:           iss,
			Scope:         scope,
			Aud:           stdAud,
			iat:           iatT,
			exp:           expT,
			PrivateClaims: testCase.input,
		}
		cJSON, err := base64Decode(c.encode())
		if err != nil {
			t.Fatalf("error decoding claim set: %v", err)
		}
		if string(cJSON) != testCase.want {
			t.Errorf("TestClaimSetWithPrivateNameEncode: enc != want in case %s", testCase.desc)
			t.Errorf("    enc = %s", cJSON)
			t.Errorf("    want = %s", testCase.want)
		}
	}
}

// Test the NewToken constructor.
func TestNewToken(t *testing.T) {
	tok := NewToken(iss, scope, privateKeyPemBytes)
	if tok.ClaimSet.Iss != iss {
		t.Error("TestNewToken: tok.ClaimSet.Iss != iss")
		t.Errorf("  tok.ClaimSet.Iss = %s", tok.ClaimSet.Iss)
		t.Errorf("               iss = %s", iss)
	}
	if tok.ClaimSet.Scope != scope {
		t.Error("TestNewToken: tok.ClaimSet.Scope != scope")
		t.Errorf("  tok.ClaimSet.Scope = %s", tok.ClaimSet.Scope)
		t.Errorf("               scope = %s", scope)
	}
	if tok.ClaimSet.Aud != stdAud {
		t.Error("TestNewToken: tok.ClaimSet.Aud != stdAud")
		t.Errorf("  tok.ClaimSet.Aud = %s", tok.ClaimSet.Aud)
		t.Errorf("            stdAud = %s", stdAud)
	}
	if !bytes.Equal(tok.Key, privateKeyPemBytes) {
		t.Error("TestNewToken: tok.Key != privateKeyPemBytes")
		t.Errorf("             tok.Key = %s", tok.Key)
		t.Errorf("  privateKeyPemBytes = %s", privateKeyPemBytes)
	}
}

// Make sure the private key parsing functions work.
func TestParsePrivateKey(t *testing.T) {
	tok := &Token{
		Key: privateKeyPemBytes,
	}
	err := tok.parsePrivateKey()
	if err != nil {
		t.Errorf("TestParsePrivateKey:tok.parsePrivateKey: %v", err)
	}
}

// Test that the token signature generated matches the golden standard.
func TestTokenSign(t *testing.T) {
	tok := &Token{
		Key:    privateKeyPemBytes,
		claim:  claimSetEnc,
		header: headerEnc,
	}
	err := tok.parsePrivateKey()
	if err != nil {
		t.Errorf("TestTokenSign:tok.parsePrivateKey: %v", err)
	}
	err = tok.sign()
	if err != nil {
		t.Errorf("TestTokenSign:tok.sign: %v", err)
	}
	if tok.sig != sigEnc {
		t.Error("TestTokenSign: tok.sig != sigEnc")
		t.Errorf("  tok.sig = %s", tok.sig)
		t.Errorf("   sigEnc = %s", sigEnc)
	}
}

// Test that the token expiration function is working.
func TestTokenExpired(t *testing.T) {
	c := &ClaimSet{}
	tok := &Token{
		ClaimSet: c,
	}
	now := time.Now()
	c.setTimes(now)
	if tok.Expired() != false {
		t.Error("TestTokenExpired: tok.Expired != false")
	}
	// Set the times as if they were set 2 hours ago.
	c.setTimes(now.Add(-2 * time.Hour))
	if tok.Expired() != true {
		t.Error("TestTokenExpired: tok.Expired != true")
	}
}

// Given a well formed Token, test for proper encoding.
func TestTokenEncode(t *testing.T) {
	c := &ClaimSet{
		Iss:   iss,
		Scope: scope,
		exp:   time.Unix(exp, 0),
		iat:   time.Unix(iat, 0),
	}
	tok := &Token{
		ClaimSet: c,
		Header:   stdHeader,
		Key:      privateKeyPemBytes,
	}
	enc, err := tok.Encode()
	if err != nil {
		t.Errorf("TestTokenEncode:tok.Assertion: %v", err)
	}
	if enc != tokEnc {
		t.Error("TestTokenEncode: enc != tokEnc")
		t.Errorf("     enc = %s", enc)
		t.Errorf("  tokEnc = %s", tokEnc)
	}
}

// Given a well formed Token we should get back a well formed request.
func TestBuildRequest(t *testing.T) {
	c := &ClaimSet{
		Iss:   iss,
		Scope: scope,
		exp:   time.Unix(exp, 0),
		iat:   time.Unix(iat, 0),
	}
	tok := &Token{
		ClaimSet: c,
		Header:   stdHeader,
		Key:      privateKeyPemBytes,
	}
	u, v, err := tok.buildRequest()
	if err != nil {
		t.Errorf("TestBuildRequest:BuildRequest: %v", err)
	}
	if u != c.Aud {
		t.Error("TestBuildRequest: u != c.Aud")
		t.Errorf("      u = %s", u)
		t.Errorf("  c.Aud = %s", c.Aud)
	}
	if v.Get("grant_type") != stdGrantType {
		t.Error("TestBuildRequest: grant_type != stdGrantType")
		t.Errorf("    grant_type = %s", v.Get("grant_type"))
		t.Errorf("  stdGrantType = %s", stdGrantType)
	}
	if v.Get("assertion") != tokEnc {
		t.Error("TestBuildRequest: assertion != tokEnc")
		t.Errorf("  assertion = %s", v.Get("assertion"))
		t.Errorf("     tokEnc = %s", tokEnc)
	}
}

// Given a well formed access request response we should get back a oauth.Token.
func TestHandleResponse(t *testing.T) {
	rb := &respBody{
		Access:    "1/8xbJqaOZXSUZbHLl5EOtu1pxz3fmmetKx9W8CV4t79M",
		Type:      "Bearer",
		ExpiresIn: 3600,
	}
	b, err := json.Marshal(rb)
	if err != nil {
		t.Errorf("TestHandleResponse:json.Marshal: %v", err)
	}
	r := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Body:       ioutil.NopCloser(bytes.NewReader(b)),
	}
	o, err := handleResponse(r)
	if err != nil {
		t.Errorf("TestHandleResponse:handleResponse: %v", err)
	}
	if o.AccessToken != rb.Access {
		t.Error("TestHandleResponse: o.AccessToken != rb.Access")
		t.Errorf("  o.AccessToken = %s", o.AccessToken)
		t.Errorf("       rb.Access = %s", rb.Access)
	}
	if o.Expired() {
		t.Error("TestHandleResponse: o.Expired == true")
	}
}

// passthrough signature for test
type FakeSigner struct{}

func (f FakeSigner) Sign(tok *Token) ([]byte, []byte, error) {
	block, _ := pem.Decode(privateKeyPemBytes)
	pKey, _ := x509.ParsePKCS1PrivateKey(block.Bytes)
	ss := headerEnc + "." + claimSetEnc
	h := sha256.New()
	h.Write([]byte(ss))
	b, _ := rsa.SignPKCS1v15(rand.Reader, pKey, crypto.SHA256, h.Sum(nil))
	return []byte(ss), b, nil
}

// Given an external signer, get back a valid and signed JWT
func TestExternalSigner(t *testing.T) {
	tok := NewSignerToken(iss, scope, FakeSigner{})
	enc, _ := tok.Encode()
	if enc != tokEnc {
		t.Errorf("TestExternalSigner: enc != tokEnc")
		t.Errorf("     enc = %s", enc)
		t.Errorf("  tokEnc = %s", tokEnc)
	}
}

func TestHandleResponseWithNewExpiry(t *testing.T) {
	rb := &respBody{
		IdToken: tokEnc,
	}
	b, err := json.Marshal(rb)
	if err != nil {
		t.Errorf("TestHandleResponse:json.Marshal: %v", err)
	}
	r := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Body:       ioutil.NopCloser(bytes.NewReader(b)),
	}
	o, err := handleResponse(r)
	if err != nil {
		t.Errorf("TestHandleResponse:handleResponse: %v", err)
	}
	if o.Expiry != time.Unix(exp, 0) {
		t.Error("TestHandleResponse: o.Expiry != exp")
		t.Errorf("  o.Expiry = %s", o.Expiry)
		t.Errorf("       exp = %s", time.Unix(exp, 0))
	}
}

// Placeholder for future Assert tests.
func TestAssert(t *testing.T) {
	// Since this method makes a call to BuildRequest, an htttp.Client, and
	// finally HandleResponse there is not much more to test.  This is here
	// as a placeholder if that changes.
}

// Benchmark for the end-to-end encoding of a well formed token.
func BenchmarkTokenEncode(b *testing.B) {
	b.StopTimer()
	c := &ClaimSet{
		Iss:   iss,
		Scope: scope,
		exp:   time.Unix(exp, 0),
		iat:   time.Unix(iat, 0),
	}
	tok := &Token{
		ClaimSet: c,
		Key:      privateKeyPemBytes,
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode()
	}
}
