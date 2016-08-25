// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jws provides encoding and decoding utilities for
// signed JWS messages.
package jws // import "golang.org/x/oauth2/jws"

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// ClaimSet contains information about the JWT signature including the
// permissions being requested (scopes), the target of the token, the issuer,
// the time the token was issued, and the lifetime of the token.
type ClaimSet struct {
	Iss   string `json:"iss"`             // email address of the client_id of the application making the access token request
	Scope string `json:"scope,omitempty"` // space-delimited list of the permissions the application requests
	Aud   string `json:"aud"`             // descriptor of the intended target of the assertion (Optional).
	Exp   int64  `json:"exp"`             // the expiration time of the assertion
	Iat   int64  `json:"iat"`             // the time the assertion was issued.
	Typ   string `json:"typ,omitempty"`   // token type (Optional).

	// Email for which the application is requesting delegated access (Optional).
	Sub string `json:"sub,omitempty"`

	// The old name of Sub. Client keeps setting Prn to be
	// complaint with legacy OAuth 2.0 providers. (Optional)
	Prn string `json:"prn,omitempty"`

	// See http://tools.ietf.org/html/draft-jones-json-web-token-10#section-4.3
	// This array is marshalled using custom code (see (c *ClaimSet) encode()).
	PrivateClaims map[string]interface{} `json:"-"`

	exp time.Time
	iat time.Time
}

func (c *ClaimSet) encode() (string, error) {
	if c.exp.IsZero() || c.iat.IsZero() {
		// Reverting time back for machines whose time is not perfectly in sync.
		// If client machine's time is in the future according
		// to Google servers, an access token will not be issued.
		now := time.Now().Add(-10 * time.Second)
		c.iat = now
		c.exp = now.Add(time.Hour)
	}

	c.Exp = c.exp.Unix()
	c.Iat = c.iat.Unix()

	b, err := json.Marshal(c)
	if err != nil {
		return "", err
	}

	if len(c.PrivateClaims) == 0 {
		return base64Encode(b), nil
	}

	// Marshal private claim set and then append it to b.
	prv, err := json.Marshal(c.PrivateClaims)
	if err != nil {
		return "", fmt.Errorf("jws: invalid map of private claims %v", c.PrivateClaims)
	}

	// Concatenate public and private claim JSON objects.
	if !bytes.HasSuffix(b, []byte{'}'}) {
		return "", fmt.Errorf("jws: invalid JSON %s", b)
	}
	if !bytes.HasPrefix(prv, []byte{'{'}) {
		return "", fmt.Errorf("jws: invalid JSON %s", prv)
	}
	b[len(b)-1] = ','         // Replace closing curly brace with a comma.
	b = append(b, prv[1:]...) // Append private claims.
	return base64Encode(b), nil
}

// Header represents the header for the signed JWS payloads.
type Header struct {
	// The algorithm used for signature.
	Algorithm string `json:"alg"`

	// Represents the token type.
	Typ string `json:"typ"`
}

func (h *Header) encode() (string, error) {
	b, err := json.Marshal(h)
	if err != nil {
		return "", err
	}
	return base64Encode(b), nil
}

// Decode decodes a claim set from a JWS payload.
func Decode(payload string) (*ClaimSet, error) {
	// decode returned id token to get expiry
	s := strings.Split(payload, ".")
	if len(s) < 2 {
		// TODO(jbd): Provide more context about the error.
		return nil, errors.New("jws: invalid token received")
	}
	decoded, err := base64Decode(s[1])
	if err != nil {
		return nil, err
	}
	c := &ClaimSet{}
	err = json.NewDecoder(bytes.NewBuffer(decoded)).Decode(c)
	return c, err
}

// Encode encodes a signed JWS with provided header and claim set.
func Encode(header *Header, c *ClaimSet, signature *rsa.PrivateKey) (string, error) {
	head, err := header.encode()
	if err != nil {
		return "", err
	}
	cs, err := c.encode()
	if err != nil {
		return "", err
	}
	ss := fmt.Sprintf("%s.%s", head, cs)
	h := sha256.New()
	h.Write([]byte(ss))
	b, err := rsa.SignPKCS1v15(rand.Reader, signature, crypto.SHA256, h.Sum(nil))
	if err != nil {
		return "", err
	}
	sig := base64Encode(b)
	return fmt.Sprintf("%s.%s", ss, sig), nil
}

// base64Encode returns and Base64url encoded version of the input string with any
// trailing "=" stripped.
func base64Encode(b []byte) string {
	return strings.TrimRight(base64.URLEncoding.EncodeToString(b), "=")
}

// base64Decode decodes the Base64url encoded string
func base64Decode(s string) ([]byte, error) {
	// add back missing padding
	switch len(s) % 4 {
	case 2:
		s += "=="
	case 3:
		s += "="
	}
	return base64.URLEncoding.DecodeString(s)
}
