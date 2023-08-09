package oidc

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"golang.org/x/oauth2"
	jose "gopkg.in/square/go-jose.v2"
)

const (
	issuerGoogleAccounts         = "https://accounts.google.com"
	issuerGoogleAccountsNoScheme = "accounts.google.com"
)

// KeySet is a set of publc JSON Web Keys that can be used to validate the signature
// of JSON web tokens. This is expected to be backed by a remote key set through
// provider metadata discovery or an in-memory set of keys delivered out-of-band.
type KeySet interface {
	// VerifySignature parses the JSON web token, verifies the signature, and returns
	// the raw payload. Header and claim fields are validated by other parts of the
	// package. For example, the KeySet does not need to check values such as signature
	// algorithm, issuer, and audience since the IDTokenVerifier validates these values
	// independently.
	//
	// If VerifySignature makes HTTP requests to verify the token, it's expected to
	// use any HTTP client associated with the context through ClientContext.
	VerifySignature(ctx context.Context, jwt string) (payload []byte, err error)
}

// IDTokenVerifier provides verification for ID Tokens.
type IDTokenVerifier struct {
	keySet KeySet
	config *Config
	issuer string
}

// NewVerifier returns a verifier manually constructed from a key set and issuer URL.
//
// It's easier to use provider discovery to construct an IDTokenVerifier than creating
// one directly. This method is intended to be used with provider that don't support
// metadata discovery, or avoiding round trips when the key set URL is already known.
//
// This constructor can be used to create a verifier directly using the issuer URL and
// JSON Web Key Set URL without using discovery:
//
//		keySet := oidc.NewRemoteKeySet(ctx, "https://www.googleapis.com/oauth2/v3/certs")
//		verifier := oidc.NewVerifier("https://accounts.google.com", keySet, config)
//
// Since KeySet is an interface, this constructor can also be used to supply custom
// public key sources. For example, if a user wanted to supply public keys out-of-band
// and hold them statically in-memory:
//
//		// Custom KeySet implementation.
//		keySet := newStatisKeySet(publicKeys...)
//
//		// Verifier uses the custom KeySet implementation.
//		verifier := oidc.NewVerifier("https://auth.example.com", keySet, config)
//
func NewVerifier(issuerURL string, keySet KeySet, config *Config) *IDTokenVerifier {
	return &IDTokenVerifier{keySet: keySet, config: config, issuer: issuerURL}
}

// Config is the configuration for an IDTokenVerifier.
type Config struct {
	// Expected audience of the token. For a majority of the cases this is expected to be
	// the ID of the client that initialized the login flow. It may occasionally differ if
	// the provider supports the authorizing party (azp) claim.
	//
	// If not provided, users must explicitly set SkipClientIDCheck.
	ClientID string
	// If specified, only this set of algorithms may be used to sign the JWT.
	//
	// Since many providers only support RS256, SupportedSigningAlgs defaults to this value.
	SupportedSigningAlgs []string

	// If true, no ClientID check performed. Must be true if ClientID field is empty.
	SkipClientIDCheck bool
	// If true, token expiry is not checked.
	SkipExpiryCheck bool

	// SkipIssuerCheck is intended for specialized cases where the the caller wishes to
	// defer issuer validation. When enabled, callers MUST independently verify the Token's
	// Issuer is a known good value.
	//
	// Mismatched issuers often indicate client mis-configuration. If mismatches are
	// unexpected, evaluate if the provided issuer URL is incorrect instead of enabling
	// this option.
	SkipIssuerCheck bool

	// Time function to check Token expiry. Defaults to time.Now
	Now func() time.Time
}

// Verifier returns an IDTokenVerifier that uses the provider's key set to verify JWTs.
//
// The returned IDTokenVerifier is tied to the Provider's context and its behavior is
// undefined once the Provider's context is canceled.
func (p *Provider) Verifier(config *Config) *IDTokenVerifier {
	return NewVerifier(p.issuer, p.remoteKeySet, config)
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

// Returns the Claims from the distributed JWT token
func resolveDistributedClaim(ctx context.Context, verifier *IDTokenVerifier, src claimSource) ([]byte, error) {
	req, err := http.NewRequest("GET", src.Endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("malformed request: %v", err)
	}
	if src.AccessToken != "" {
		req.Header.Set("Authorization", "Bearer "+src.AccessToken)
	}

	resp, err := doRequest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("oidc: Request to endpoint failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("unable to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("oidc: request failed: %v", resp.StatusCode)
	}

	token, err := verifier.Verify(ctx, string(body))
	if err != nil {
		return nil, fmt.Errorf("malformed response body: %v", err)
	}

	return token.claims, nil
}

func parseClaim(raw []byte, name string, v interface{}) error {
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return err
	}

	val, ok := parsed[name]
	if !ok {
		return fmt.Errorf("claim doesn't exist: %s", name)
	}

	return json.Unmarshal([]byte(val), v)
}

// Verify parses a raw ID Token, verifies it's been signed by the provider, preforms
// any additional checks depending on the Config, and returns the payload.
//
// Verify does NOT do nonce validation, which is the callers responsibility.
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
		return nil, fmt.Errorf("oidc: malformed jwt: %v", err)
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

	distributedClaims := make(map[string]claimSource)

	//step through the token to map claim names to claim sources"
	for cn, src := range token.ClaimNames {
		if src == "" {
			return nil, fmt.Errorf("oidc: failed to obtain source from claim name")
		}
		s, ok := token.ClaimSources[src]
		if !ok {
			return nil, fmt.Errorf("oidc: source does not exist")
		}
		distributedClaims[cn] = s
	}

	t := &IDToken{
		Issuer:            token.Issuer,
		Subject:           token.Subject,
		Audience:          []string(token.Audience),
		Expiry:            time.Time(token.Expiry),
		IssuedAt:          time.Time(token.IssuedAt),
		Nonce:             token.Nonce,
		AccessTokenHash:   token.AtHash,
		claims:            payload,
		distributedClaims: distributedClaims,
	}

	// Check issuer.
	if !v.config.SkipIssuerCheck && t.Issuer != v.issuer {
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
			return nil, fmt.Errorf("oidc: invalid configuration, clientID must be provided or SkipClientIDCheck must be set")
		}
	}

	// If a SkipExpiryCheck is false, make sure token is not expired.
	if !v.config.SkipExpiryCheck {
		now := time.Now
		if v.config.Now != nil {
			now = v.config.Now
		}
		nowTime := now()

		if t.Expiry.Before(nowTime) {
			return nil, fmt.Errorf("oidc: token is expired (Token Expiry: %v)", t.Expiry)
		}

		// If nbf claim is provided in token, ensure that it is indeed in the past.
		if token.NotBefore != nil {
			nbfTime := time.Time(*token.NotBefore)
			leeway := 1 * time.Minute

			if nowTime.Add(leeway).Before(nbfTime) {
				return nil, fmt.Errorf("oidc: current time %v before the nbf (not before) time: %v", nowTime, nbfTime)
			}
		}
	}

	switch len(jws.Signatures) {
	case 0:
		return nil, fmt.Errorf("oidc: id token not signed")
	case 1:
	default:
		return nil, fmt.Errorf("oidc: multiple signatures on id token not supported")
	}

	sig := jws.Signatures[0]
	supportedSigAlgs := v.config.SupportedSigningAlgs
	if len(supportedSigAlgs) == 0 {
		supportedSigAlgs = []string{RS256}
	}

	if !contains(supportedSigAlgs, sig.Header.Algorithm) {
		return nil, fmt.Errorf("oidc: id token signed with unsupported algorithm, expected %q got %q", supportedSigAlgs, sig.Header.Algorithm)
	}

	t.sigAlgorithm = sig.Header.Algorithm

	gotPayload, err := v.keySet.VerifySignature(ctx, rawIDToken)
	if err != nil {
		return nil, fmt.Errorf("failed to verify signature: %v", err)
	}

	// Ensure that the payload returned by the square actually matches the payload parsed earlier.
	if !bytes.Equal(gotPayload, payload) {
		return nil, errors.New("oidc: internal error, payload parsed did not match previous payload")
	}

	return t, nil
}

// Nonce returns an auth code option which requires the ID Token created by the
// OpenID Connect provider to contain the specified nonce.
func Nonce(nonce string) oauth2.AuthCodeOption {
	return oauth2.SetAuthURLParam("nonce", nonce)
}
