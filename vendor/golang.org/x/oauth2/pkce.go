// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth2

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"net/url"
)

const (
	codeChallengeKey       = "code_challenge"
	codeChallengeMethodKey = "code_challenge_method"
	codeVerifierKey        = "code_verifier"
)

// GenerateVerifier generates a PKCE code verifier with 32 octets of randomness.
// This follows recommendations in RFC 7636.
//
// A fresh verifier should be generated for each authorization.
// The resulting verifier should be passed to [Config.AuthCodeURL] or [Config.DeviceAuth]
// with [S256ChallengeOption], and to [Config.Exchange] or [Config.DeviceAccessToken]
// with [VerifierOption].
func GenerateVerifier() string {
	// "RECOMMENDED that the output of a suitable random number generator be
	// used to create a 32-octet sequence.  The octet sequence is then
	// base64url-encoded to produce a 43-octet URL-safe string to use as the
	// code verifier."
	// https://datatracker.ietf.org/doc/html/rfc7636#section-4.1
	data := make([]byte, 32)
	if _, err := rand.Read(data); err != nil {
		panic(err)
	}
	return base64.RawURLEncoding.EncodeToString(data)
}

// VerifierOption returns a PKCE code verifier [AuthCodeOption]. It should only be
// passed to [Config.Exchange] or [Config.DeviceAccessToken].
func VerifierOption(verifier string) AuthCodeOption {
	return setParam{k: codeVerifierKey, v: verifier}
}

// S256ChallengeFromVerifier returns a PKCE code challenge derived from verifier with method S256.
//
// Prefer to use [S256ChallengeOption] where possible.
func S256ChallengeFromVerifier(verifier string) string {
	sha := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(sha[:])
}

// S256ChallengeOption derives a PKCE code challenge from the verifier with
// method S256. It should be passed to [Config.AuthCodeURL] or [Config.DeviceAuth]
// only.
func S256ChallengeOption(verifier string) AuthCodeOption {
	return challengeOption{
		challenge_method: "S256",
		challenge:        S256ChallengeFromVerifier(verifier),
	}
}

type challengeOption struct{ challenge_method, challenge string }

func (p challengeOption) setValue(m url.Values) {
	m.Set(codeChallengeMethodKey, p.challenge_method)
	m.Set(codeChallengeKey, p.challenge)
}
