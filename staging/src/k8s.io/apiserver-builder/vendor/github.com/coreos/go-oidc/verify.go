package oidc

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"golang.org/x/oauth2"
	jose "gopkg.in/square/go-jose.v2"
)

const (
	issuerGoogleAccounts         = "https://accounts.google.com"
	issuerGoogleAccountsNoScheme = "accounts.google.com"
)

// IDTokenVerifier provides verification for ID Tokens.
type IDTokenVerifier struct {
	keySet *remoteKeySet
	config *Config
	issuer string
}

// Config is the configuration for an IDTokenVerifier.
type Config struct {
	// Expected audience of the token. For a majority of the cases this is expected to be
	// the ID of the client that initialized the login flow. It may occasionally differ if
	// the provider supports the authorizing party (azp) claim.
	//
	// If not provided, users must explicitly set SkipClientIDCheck.
	ClientID string
	// Method to verify the ID Token nonce. If a nonce is present and this method
	// is nil, users must explicitly set SkipNonceCheck.
	//
	// If the ID Token nonce is empty, for example if the client didn't provide a nonce in
	// the initial redirect, this may be nil.
	ClaimNonce func(nonce string) error
	// If specified, only this set of algorithms may be used to sign the JWT.
	//
	// Since many providers only support RS256, SupportedSigningAlgs defaults to this value.
	SupportedSigningAlgs []string

	// If true, no ClientID check performed. Must be true if ClientID field is empty.
	SkipClientIDCheck bool
	// If true, token expiry is not checked.
	SkipExpiryCheck bool
	// If true, nonce claim is not checked. Must be true if ClaimNonce field is empty.
	SkipNonceCheck bool

	// Time function to check Token expiry. Defaults to time.Now
	Now func() time.Time
}

// Verifier returns an IDTokenVerifier that uses the provider's key set to verify JWTs.
//
// The returned IDTokenVerifier is tied to the Provider's context and its behavior is
// undefined once the Provider's context is canceled.
func (p *Provider) Verifier(config *Config) *IDTokenVerifier {

	return newVerifier(p.remoteKeySet, config, p.issuer)
}

func newVerifier(keySet *remoteKeySet, config *Config, issuer string) *IDTokenVerifier {
	// If SupportedSigningAlgs is empty defaults to only support RS256.
	if len(config.SupportedSigningAlgs) == 0 {
		config.SupportedSigningAlgs = []string{RS256}
	}

	return &IDTokenVerifier{
		keySet: keySet,
		config: config,
		issuer: issuer,
	}
}

func parseJWT(p string) ([]byte, error) {
	parts := strings.Split(p, ".")
	if len(parts) < 2 {
		return nil, fmt.Errorf("oidc: malformed jwt, expected 3 parts got %d", len(parts))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("oidc: malformed jwt payload: %v", err)
	}
	return payload, nil
}

func contains(sli []string, ele string) bool {
	for _, s := range sli {
		if s == ele {
			return true
		}
	}
	return false
}

// Verify parses a raw ID Token, verifies it's been signed by the provider, preforms
// any additional checks depending on the Config, and returns the payload.
//
// See: https://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation
//
//    oauth2Token, err := oauth2Config.Exchange(ctx, r.URL.Query().Get("code"))
//    if err != nil {
//        // handle error
//    }
//
//    // Extract the ID Token from oauth2 token.
//    rawIDToken, ok := oauth2Token.Extra("id_token").(string)
//    if !ok {
//        // handle error
//    }
//
//    token, err := verifier.Verify(ctx, rawIDToken)
//
func (v *IDTokenVerifier) Verify(ctx context.Context, rawIDToken string) (*IDToken, error) {
	jws, err := jose.ParseSigned(rawIDToken)
	if err != nil {
		return nil, fmt.Errorf("oidc: mallformed jwt: %v", err)
	}

	// Throw out tokens with invalid claims before trying to verify the token. This lets
	// us do cheap checks before possibly re-syncing keys.
	payload, err := parseJWT(rawIDToken)
	if err != nil {
		return nil, fmt.Errorf("oidc: malformed jwt: %v", err)
	}
	var token idToken
	if err := json.Unmarshal(payload, &token); err != nil {
		return nil, fmt.Errorf("oidc: failed to unmarshal claims: %v", err)
	}

	t := &IDToken{
		Issuer:   token.Issuer,
		Subject:  token.Subject,
		Audience: []string(token.Audience),
		Expiry:   time.Time(token.Expiry),
		IssuedAt: time.Time(token.IssuedAt),
		Nonce:    token.Nonce,
		claims:   payload,
	}

	// Check issuer.
	if t.Issuer != v.issuer {
		// Google sometimes returns "accounts.google.com" as the issuer claim instead of
		// the required "https://accounts.google.com". Detect this case and allow it only
		// for Google.
		//
		// We will not add hooks to let other providers go off spec like this.
		if !(v.issuer == issuerGoogleAccounts && t.Issuer == issuerGoogleAccountsNoScheme) {
			return nil, fmt.Errorf("oidc: id token issued by a different provider, expected %q got %q", v.issuer, t.Issuer)
		}
	}

	// If a client ID has been provided, make sure it's part of the audience. SkipClientIDCheck must be true if ClientID is empty.
	//
	// This check DOES NOT ensure that the ClientID is the party to which the ID Token was issued (i.e. Authorized party).
	if !v.config.SkipClientIDCheck {
		if v.config.ClientID != "" {
			if !contains(t.Audience, v.config.ClientID) {
				return nil, fmt.Errorf("oidc: expected audience %q got %q", v.config.ClientID, t.Audience)
			}
		} else {
			return nil, fmt.Errorf("oidc: Invalid configuration. ClientID must be provided or SkipClientIDCheck must be set.")
		}
	}

	// If a SkipExpiryCheck is false, make sure token is not expired.
	if !v.config.SkipExpiryCheck {
		now := time.Now
		if v.config.Now != nil {
			now = v.config.Now
		}

		if t.Expiry.Before(now()) {
			return nil, fmt.Errorf("oidc: token is expired (Token Expiry: %v)", t.Expiry)
		}
	}

	// If a set of required algorithms has been provided, ensure that the signatures use those.
	var keyIDs, gotAlgs []string
	for _, sig := range jws.Signatures {
		if len(v.config.SupportedSigningAlgs) == 0 || contains(v.config.SupportedSigningAlgs, sig.Header.Algorithm) {
			keyIDs = append(keyIDs, sig.Header.KeyID)
		} else {
			gotAlgs = append(gotAlgs, sig.Header.Algorithm)
		}
	}
	if len(keyIDs) == 0 {
		return nil, fmt.Errorf("oidc: no signatures use a supported algorithm, expected %q got %q", v.config.SupportedSigningAlgs, gotAlgs)
	}

	// Get keys from the remote key set. This may trigger a re-sync.
	keys, err := v.keySet.keysWithID(ctx, keyIDs)
	if err != nil {
		return nil, fmt.Errorf("oidc: get keys for id token: %v", err)
	}
	if len(keys) == 0 {
		return nil, fmt.Errorf("oidc: no keys match signature ID(s) %q", keyIDs)
	}

	// Try to use a key to validate the signature.
	var gotPayload []byte
	for _, key := range keys {
		if p, err := jws.Verify(&key); err == nil {
			gotPayload = p
		}
	}
	if len(gotPayload) == 0 {
		return nil, fmt.Errorf("oidc: failed to verify id token")
	}

	// Ensure that the payload returned by the square actually matches the payload parsed earlier.
	if !bytes.Equal(gotPayload, payload) {
		return nil, errors.New("oidc: internal error, payload parsed did not match previous payload")
	}

	// Check the nonce after we've verified the token. We don't want to allow unverified
	// payloads to trigger a nonce lookup.
	// If SkipNonceCheck is not set ClaimNonce cannot be Nil.
	if !v.config.SkipNonceCheck && t.Nonce != "" {
		if v.config.ClaimNonce != nil {
			if err := v.config.ClaimNonce(t.Nonce); err != nil {
				return nil, err
			}
		} else {
			return nil, fmt.Errorf("oidc: Invalid configuration. ClaimNonce must be provided or SkipNonceCheck must be set.")
		}
	}

	return t, nil
}

// Nonce returns an auth code option which requires the ID Token created by the
// OpenID Connect provider to contain the specified nonce.
func Nonce(nonce string) oauth2.AuthCodeOption {
	return oauth2.SetAuthURLParam("nonce", nonce)
}
