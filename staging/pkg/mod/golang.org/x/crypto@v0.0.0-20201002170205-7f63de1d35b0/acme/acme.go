// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package acme provides an implementation of the
// Automatic Certificate Management Environment (ACME) spec.
// The intial implementation was based on ACME draft-02 and
// is now being extended to comply with RFC 8555.
// See https://tools.ietf.org/html/draft-ietf-acme-acme-02
// and https://tools.ietf.org/html/rfc8555 for details.
//
// Most common scenarios will want to use autocert subdirectory instead,
// which provides automatic access to certificates from Let's Encrypt
// and any other ACME-based CA.
//
// This package is a work in progress and makes no API stability promises.
package acme

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	// LetsEncryptURL is the Directory endpoint of Let's Encrypt CA.
	LetsEncryptURL = "https://acme-v02.api.letsencrypt.org/directory"

	// ALPNProto is the ALPN protocol name used by a CA server when validating
	// tls-alpn-01 challenges.
	//
	// Package users must ensure their servers can negotiate the ACME ALPN in
	// order for tls-alpn-01 challenge verifications to succeed.
	// See the crypto/tls package's Config.NextProtos field.
	ALPNProto = "acme-tls/1"
)

// idPeACMEIdentifier is the OID for the ACME extension for the TLS-ALPN challenge.
// https://tools.ietf.org/html/draft-ietf-acme-tls-alpn-05#section-5.1
var idPeACMEIdentifier = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 1, 31}

const (
	maxChainLen = 5       // max depth and breadth of a certificate chain
	maxCertSize = 1 << 20 // max size of a certificate, in DER bytes
	// Used for decoding certs from application/pem-certificate-chain response,
	// the default when in RFC mode.
	maxCertChainSize = maxCertSize * maxChainLen

	// Max number of collected nonces kept in memory.
	// Expect usual peak of 1 or 2.
	maxNonces = 100
)

// Client is an ACME client.
// The only required field is Key. An example of creating a client with a new key
// is as follows:
//
// 	key, err := rsa.GenerateKey(rand.Reader, 2048)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	client := &Client{Key: key}
//
type Client struct {
	// Key is the account key used to register with a CA and sign requests.
	// Key.Public() must return a *rsa.PublicKey or *ecdsa.PublicKey.
	//
	// The following algorithms are supported:
	// RS256, ES256, ES384 and ES512.
	// See RFC7518 for more details about the algorithms.
	Key crypto.Signer

	// HTTPClient optionally specifies an HTTP client to use
	// instead of http.DefaultClient.
	HTTPClient *http.Client

	// DirectoryURL points to the CA directory endpoint.
	// If empty, LetsEncryptURL is used.
	// Mutating this value after a successful call of Client's Discover method
	// will have no effect.
	DirectoryURL string

	// RetryBackoff computes the duration after which the nth retry of a failed request
	// should occur. The value of n for the first call on failure is 1.
	// The values of r and resp are the request and response of the last failed attempt.
	// If the returned value is negative or zero, no more retries are done and an error
	// is returned to the caller of the original method.
	//
	// Requests which result in a 4xx client error are not retried,
	// except for 400 Bad Request due to "bad nonce" errors and 429 Too Many Requests.
	//
	// If RetryBackoff is nil, a truncated exponential backoff algorithm
	// with the ceiling of 10 seconds is used, where each subsequent retry n
	// is done after either ("Retry-After" + jitter) or (2^n seconds + jitter),
	// preferring the former if "Retry-After" header is found in the resp.
	// The jitter is a random value up to 1 second.
	RetryBackoff func(n int, r *http.Request, resp *http.Response) time.Duration

	// UserAgent is prepended to the User-Agent header sent to the ACME server,
	// which by default is this package's name and version.
	//
	// Reusable libraries and tools in particular should set this value to be
	// identifiable by the server, in case they are causing issues.
	UserAgent string

	cacheMu sync.Mutex
	dir     *Directory // cached result of Client's Discover method
	kid     keyID      // cached Account.URI obtained from registerRFC or getAccountRFC

	noncesMu sync.Mutex
	nonces   map[string]struct{} // nonces collected from previous responses
}

// accountKID returns a key ID associated with c.Key, the account identity
// provided by the CA during RFC based registration.
// It assumes c.Discover has already been called.
//
// accountKID requires at most one network roundtrip.
// It caches only successful result.
//
// When in pre-RFC mode or when c.getRegRFC responds with an error, accountKID
// returns noKeyID.
func (c *Client) accountKID(ctx context.Context) keyID {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()
	if !c.dir.rfcCompliant() {
		return noKeyID
	}
	if c.kid != noKeyID {
		return c.kid
	}
	a, err := c.getRegRFC(ctx)
	if err != nil {
		return noKeyID
	}
	c.kid = keyID(a.URI)
	return c.kid
}

// Discover performs ACME server discovery using c.DirectoryURL.
//
// It caches successful result. So, subsequent calls will not result in
// a network round-trip. This also means mutating c.DirectoryURL after successful call
// of this method will have no effect.
func (c *Client) Discover(ctx context.Context) (Directory, error) {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()
	if c.dir != nil {
		return *c.dir, nil
	}

	res, err := c.get(ctx, c.directoryURL(), wantStatus(http.StatusOK))
	if err != nil {
		return Directory{}, err
	}
	defer res.Body.Close()
	c.addNonce(res.Header)

	var v struct {
		Reg          string `json:"new-reg"`
		RegRFC       string `json:"newAccount"`
		Authz        string `json:"new-authz"`
		AuthzRFC     string `json:"newAuthz"`
		OrderRFC     string `json:"newOrder"`
		Cert         string `json:"new-cert"`
		Revoke       string `json:"revoke-cert"`
		RevokeRFC    string `json:"revokeCert"`
		NonceRFC     string `json:"newNonce"`
		KeyChangeRFC string `json:"keyChange"`
		Meta         struct {
			Terms           string   `json:"terms-of-service"`
			TermsRFC        string   `json:"termsOfService"`
			WebsiteRFC      string   `json:"website"`
			CAA             []string `json:"caa-identities"`
			CAARFC          []string `json:"caaIdentities"`
			ExternalAcctRFC bool     `json:"externalAccountRequired"`
		}
	}
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return Directory{}, err
	}
	if v.OrderRFC == "" {
		// Non-RFC compliant ACME CA.
		c.dir = &Directory{
			RegURL:    v.Reg,
			AuthzURL:  v.Authz,
			CertURL:   v.Cert,
			RevokeURL: v.Revoke,
			Terms:     v.Meta.Terms,
			Website:   v.Meta.WebsiteRFC,
			CAA:       v.Meta.CAA,
		}
		return *c.dir, nil
	}
	// RFC compliant ACME CA.
	c.dir = &Directory{
		RegURL:                  v.RegRFC,
		AuthzURL:                v.AuthzRFC,
		OrderURL:                v.OrderRFC,
		RevokeURL:               v.RevokeRFC,
		NonceURL:                v.NonceRFC,
		KeyChangeURL:            v.KeyChangeRFC,
		Terms:                   v.Meta.TermsRFC,
		Website:                 v.Meta.WebsiteRFC,
		CAA:                     v.Meta.CAARFC,
		ExternalAccountRequired: v.Meta.ExternalAcctRFC,
	}
	return *c.dir, nil
}

func (c *Client) directoryURL() string {
	if c.DirectoryURL != "" {
		return c.DirectoryURL
	}
	return LetsEncryptURL
}

// CreateCert requests a new certificate using the Certificate Signing Request csr encoded in DER format.
// It is incompatible with RFC 8555. Callers should use CreateOrderCert when interfacing
// with an RFC-compliant CA.
//
// The exp argument indicates the desired certificate validity duration. CA may issue a certificate
// with a different duration.
// If the bundle argument is true, the returned value will also contain the CA (issuer) certificate chain.
//
// In the case where CA server does not provide the issued certificate in the response,
// CreateCert will poll certURL using c.FetchCert, which will result in additional round-trips.
// In such a scenario, the caller can cancel the polling with ctx.
//
// CreateCert returns an error if the CA's response or chain was unreasonably large.
// Callers are encouraged to parse the returned value to ensure the certificate is valid and has the expected features.
func (c *Client) CreateCert(ctx context.Context, csr []byte, exp time.Duration, bundle bool) (der [][]byte, certURL string, err error) {
	if _, err := c.Discover(ctx); err != nil {
		return nil, "", err
	}

	req := struct {
		Resource  string `json:"resource"`
		CSR       string `json:"csr"`
		NotBefore string `json:"notBefore,omitempty"`
		NotAfter  string `json:"notAfter,omitempty"`
	}{
		Resource: "new-cert",
		CSR:      base64.RawURLEncoding.EncodeToString(csr),
	}
	now := timeNow()
	req.NotBefore = now.Format(time.RFC3339)
	if exp > 0 {
		req.NotAfter = now.Add(exp).Format(time.RFC3339)
	}

	res, err := c.post(ctx, nil, c.dir.CertURL, req, wantStatus(http.StatusCreated))
	if err != nil {
		return nil, "", err
	}
	defer res.Body.Close()

	curl := res.Header.Get("Location") // cert permanent URL
	if res.ContentLength == 0 {
		// no cert in the body; poll until we get it
		cert, err := c.FetchCert(ctx, curl, bundle)
		return cert, curl, err
	}
	// slurp issued cert and CA chain, if requested
	cert, err := c.responseCert(ctx, res, bundle)
	return cert, curl, err
}

// FetchCert retrieves already issued certificate from the given url, in DER format.
// It retries the request until the certificate is successfully retrieved,
// context is cancelled by the caller or an error response is received.
//
// If the bundle argument is true, the returned value also contains the CA (issuer)
// certificate chain.
//
// FetchCert returns an error if the CA's response or chain was unreasonably large.
// Callers are encouraged to parse the returned value to ensure the certificate is valid
// and has expected features.
func (c *Client) FetchCert(ctx context.Context, url string, bundle bool) ([][]byte, error) {
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}
	if dir.rfcCompliant() {
		return c.fetchCertRFC(ctx, url, bundle)
	}

	// Legacy non-authenticated GET request.
	res, err := c.get(ctx, url, wantStatus(http.StatusOK))
	if err != nil {
		return nil, err
	}
	return c.responseCert(ctx, res, bundle)
}

// RevokeCert revokes a previously issued certificate cert, provided in DER format.
//
// The key argument, used to sign the request, must be authorized
// to revoke the certificate. It's up to the CA to decide which keys are authorized.
// For instance, the key pair of the certificate may be authorized.
// If the key is nil, c.Key is used instead.
func (c *Client) RevokeCert(ctx context.Context, key crypto.Signer, cert []byte, reason CRLReasonCode) error {
	dir, err := c.Discover(ctx)
	if err != nil {
		return err
	}
	if dir.rfcCompliant() {
		return c.revokeCertRFC(ctx, key, cert, reason)
	}

	// Legacy CA.
	body := &struct {
		Resource string `json:"resource"`
		Cert     string `json:"certificate"`
		Reason   int    `json:"reason"`
	}{
		Resource: "revoke-cert",
		Cert:     base64.RawURLEncoding.EncodeToString(cert),
		Reason:   int(reason),
	}
	res, err := c.post(ctx, key, dir.RevokeURL, body, wantStatus(http.StatusOK))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return nil
}

// AcceptTOS always returns true to indicate the acceptance of a CA's Terms of Service
// during account registration. See Register method of Client for more details.
func AcceptTOS(tosURL string) bool { return true }

// Register creates a new account with the CA using c.Key.
// It returns the registered account. The account acct is not modified.
//
// The registration may require the caller to agree to the CA's Terms of Service (TOS).
// If so, and the account has not indicated the acceptance of the terms (see Account for details),
// Register calls prompt with a TOS URL provided by the CA. Prompt should report
// whether the caller agrees to the terms. To always accept the terms, the caller can use AcceptTOS.
//
// When interfacing with an RFC-compliant CA, non-RFC 8555 fields of acct are ignored
// and prompt is called if Directory's Terms field is non-zero.
// Also see Error's Instance field for when a CA requires already registered accounts to agree
// to an updated Terms of Service.
func (c *Client) Register(ctx context.Context, acct *Account, prompt func(tosURL string) bool) (*Account, error) {
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}
	if dir.rfcCompliant() {
		return c.registerRFC(ctx, acct, prompt)
	}

	// Legacy ACME draft registration flow.
	a, err := c.doReg(ctx, dir.RegURL, "new-reg", acct)
	if err != nil {
		return nil, err
	}
	var accept bool
	if a.CurrentTerms != "" && a.CurrentTerms != a.AgreedTerms {
		accept = prompt(a.CurrentTerms)
	}
	if accept {
		a.AgreedTerms = a.CurrentTerms
		a, err = c.UpdateReg(ctx, a)
	}
	return a, err
}

// GetReg retrieves an existing account associated with c.Key.
//
// The url argument is an Account URI used with pre-RFC 8555 CAs.
// It is ignored when interfacing with an RFC-compliant CA.
func (c *Client) GetReg(ctx context.Context, url string) (*Account, error) {
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}
	if dir.rfcCompliant() {
		return c.getRegRFC(ctx)
	}

	// Legacy CA.
	a, err := c.doReg(ctx, url, "reg", nil)
	if err != nil {
		return nil, err
	}
	a.URI = url
	return a, nil
}

// UpdateReg updates an existing registration.
// It returns an updated account copy. The provided account is not modified.
//
// When interfacing with RFC-compliant CAs, a.URI is ignored and the account URL
// associated with c.Key is used instead.
func (c *Client) UpdateReg(ctx context.Context, acct *Account) (*Account, error) {
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}
	if dir.rfcCompliant() {
		return c.updateRegRFC(ctx, acct)
	}

	// Legacy CA.
	uri := acct.URI
	a, err := c.doReg(ctx, uri, "reg", acct)
	if err != nil {
		return nil, err
	}
	a.URI = uri
	return a, nil
}

// Authorize performs the initial step in the pre-authorization flow,
// as opposed to order-based flow.
// The caller will then need to choose from and perform a set of returned
// challenges using c.Accept in order to successfully complete authorization.
//
// Once complete, the caller can use AuthorizeOrder which the CA
// should provision with the already satisfied authorization.
// For pre-RFC CAs, the caller can proceed directly to requesting a certificate
// using CreateCert method.
//
// If an authorization has been previously granted, the CA may return
// a valid authorization which has its Status field set to StatusValid.
//
// More about pre-authorization can be found at
// https://tools.ietf.org/html/rfc8555#section-7.4.1.
func (c *Client) Authorize(ctx context.Context, domain string) (*Authorization, error) {
	return c.authorize(ctx, "dns", domain)
}

// AuthorizeIP is the same as Authorize but requests IP address authorization.
// Clients which successfully obtain such authorization may request to issue
// a certificate for IP addresses.
//
// See the ACME spec extension for more details about IP address identifiers:
// https://tools.ietf.org/html/draft-ietf-acme-ip.
func (c *Client) AuthorizeIP(ctx context.Context, ipaddr string) (*Authorization, error) {
	return c.authorize(ctx, "ip", ipaddr)
}

func (c *Client) authorize(ctx context.Context, typ, val string) (*Authorization, error) {
	if _, err := c.Discover(ctx); err != nil {
		return nil, err
	}

	type authzID struct {
		Type  string `json:"type"`
		Value string `json:"value"`
	}
	req := struct {
		Resource   string  `json:"resource"`
		Identifier authzID `json:"identifier"`
	}{
		Resource:   "new-authz",
		Identifier: authzID{Type: typ, Value: val},
	}
	res, err := c.post(ctx, nil, c.dir.AuthzURL, req, wantStatus(http.StatusCreated))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	var v wireAuthz
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("acme: invalid response: %v", err)
	}
	if v.Status != StatusPending && v.Status != StatusValid {
		return nil, fmt.Errorf("acme: unexpected status: %s", v.Status)
	}
	return v.authorization(res.Header.Get("Location")), nil
}

// GetAuthorization retrieves an authorization identified by the given URL.
//
// If a caller needs to poll an authorization until its status is final,
// see the WaitAuthorization method.
func (c *Client) GetAuthorization(ctx context.Context, url string) (*Authorization, error) {
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}

	var res *http.Response
	if dir.rfcCompliant() {
		res, err = c.postAsGet(ctx, url, wantStatus(http.StatusOK))
	} else {
		res, err = c.get(ctx, url, wantStatus(http.StatusOK, http.StatusAccepted))
	}
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	var v wireAuthz
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("acme: invalid response: %v", err)
	}
	return v.authorization(url), nil
}

// RevokeAuthorization relinquishes an existing authorization identified
// by the given URL.
// The url argument is an Authorization.URI value.
//
// If successful, the caller will be required to obtain a new authorization
// using the Authorize or AuthorizeOrder methods before being able to request
// a new certificate for the domain associated with the authorization.
//
// It does not revoke existing certificates.
func (c *Client) RevokeAuthorization(ctx context.Context, url string) error {
	// Required for c.accountKID() when in RFC mode.
	if _, err := c.Discover(ctx); err != nil {
		return err
	}

	req := struct {
		Resource string `json:"resource"`
		Status   string `json:"status"`
		Delete   bool   `json:"delete"`
	}{
		Resource: "authz",
		Status:   "deactivated",
		Delete:   true,
	}
	res, err := c.post(ctx, nil, url, req, wantStatus(http.StatusOK))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return nil
}

// WaitAuthorization polls an authorization at the given URL
// until it is in one of the final states, StatusValid or StatusInvalid,
// the ACME CA responded with a 4xx error code, or the context is done.
//
// It returns a non-nil Authorization only if its Status is StatusValid.
// In all other cases WaitAuthorization returns an error.
// If the Status is StatusInvalid, the returned error is of type *AuthorizationError.
func (c *Client) WaitAuthorization(ctx context.Context, url string) (*Authorization, error) {
	// Required for c.accountKID() when in RFC mode.
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}
	getfn := c.postAsGet
	if !dir.rfcCompliant() {
		getfn = c.get
	}

	for {
		res, err := getfn(ctx, url, wantStatus(http.StatusOK, http.StatusAccepted))
		if err != nil {
			return nil, err
		}

		var raw wireAuthz
		err = json.NewDecoder(res.Body).Decode(&raw)
		res.Body.Close()
		switch {
		case err != nil:
			// Skip and retry.
		case raw.Status == StatusValid:
			return raw.authorization(url), nil
		case raw.Status == StatusInvalid:
			return nil, raw.error(url)
		}

		// Exponential backoff is implemented in c.get above.
		// This is just to prevent continuously hitting the CA
		// while waiting for a final authorization status.
		d := retryAfter(res.Header.Get("Retry-After"))
		if d == 0 {
			// Given that the fastest challenges TLS-SNI and HTTP-01
			// require a CA to make at least 1 network round trip
			// and most likely persist a challenge state,
			// this default delay seems reasonable.
			d = time.Second
		}
		t := time.NewTimer(d)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, ctx.Err()
		case <-t.C:
			// Retry.
		}
	}
}

// GetChallenge retrieves the current status of an challenge.
//
// A client typically polls a challenge status using this method.
func (c *Client) GetChallenge(ctx context.Context, url string) (*Challenge, error) {
	// Required for c.accountKID() when in RFC mode.
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}

	getfn := c.postAsGet
	if !dir.rfcCompliant() {
		getfn = c.get
	}
	res, err := getfn(ctx, url, wantStatus(http.StatusOK, http.StatusAccepted))
	if err != nil {
		return nil, err
	}

	defer res.Body.Close()
	v := wireChallenge{URI: url}
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("acme: invalid response: %v", err)
	}
	return v.challenge(), nil
}

// Accept informs the server that the client accepts one of its challenges
// previously obtained with c.Authorize.
//
// The server will then perform the validation asynchronously.
func (c *Client) Accept(ctx context.Context, chal *Challenge) (*Challenge, error) {
	// Required for c.accountKID() when in RFC mode.
	dir, err := c.Discover(ctx)
	if err != nil {
		return nil, err
	}

	var req interface{} = json.RawMessage("{}") // RFC-compliant CA
	if !dir.rfcCompliant() {
		auth, err := keyAuth(c.Key.Public(), chal.Token)
		if err != nil {
			return nil, err
		}
		req = struct {
			Resource string `json:"resource"`
			Type     string `json:"type"`
			Auth     string `json:"keyAuthorization"`
		}{
			Resource: "challenge",
			Type:     chal.Type,
			Auth:     auth,
		}
	}
	res, err := c.post(ctx, nil, chal.URI, req, wantStatus(
		http.StatusOK,       // according to the spec
		http.StatusAccepted, // Let's Encrypt: see https://goo.gl/WsJ7VT (acme-divergences.md)
	))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	var v wireChallenge
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("acme: invalid response: %v", err)
	}
	return v.challenge(), nil
}

// DNS01ChallengeRecord returns a DNS record value for a dns-01 challenge response.
// A TXT record containing the returned value must be provisioned under
// "_acme-challenge" name of the domain being validated.
//
// The token argument is a Challenge.Token value.
func (c *Client) DNS01ChallengeRecord(token string) (string, error) {
	ka, err := keyAuth(c.Key.Public(), token)
	if err != nil {
		return "", err
	}
	b := sha256.Sum256([]byte(ka))
	return base64.RawURLEncoding.EncodeToString(b[:]), nil
}

// HTTP01ChallengeResponse returns the response for an http-01 challenge.
// Servers should respond with the value to HTTP requests at the URL path
// provided by HTTP01ChallengePath to validate the challenge and prove control
// over a domain name.
//
// The token argument is a Challenge.Token value.
func (c *Client) HTTP01ChallengeResponse(token string) (string, error) {
	return keyAuth(c.Key.Public(), token)
}

// HTTP01ChallengePath returns the URL path at which the response for an http-01 challenge
// should be provided by the servers.
// The response value can be obtained with HTTP01ChallengeResponse.
//
// The token argument is a Challenge.Token value.
func (c *Client) HTTP01ChallengePath(token string) string {
	return "/.well-known/acme-challenge/" + token
}

// TLSSNI01ChallengeCert creates a certificate for TLS-SNI-01 challenge response.
//
// Deprecated: This challenge type is unused in both draft-02 and RFC versions of ACME spec.
func (c *Client) TLSSNI01ChallengeCert(token string, opt ...CertOption) (cert tls.Certificate, name string, err error) {
	ka, err := keyAuth(c.Key.Public(), token)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	b := sha256.Sum256([]byte(ka))
	h := hex.EncodeToString(b[:])
	name = fmt.Sprintf("%s.%s.acme.invalid", h[:32], h[32:])
	cert, err = tlsChallengeCert([]string{name}, opt)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	return cert, name, nil
}

// TLSSNI02ChallengeCert creates a certificate for TLS-SNI-02 challenge response.
//
// Deprecated: This challenge type is unused in both draft-02 and RFC versions of ACME spec.
func (c *Client) TLSSNI02ChallengeCert(token string, opt ...CertOption) (cert tls.Certificate, name string, err error) {
	b := sha256.Sum256([]byte(token))
	h := hex.EncodeToString(b[:])
	sanA := fmt.Sprintf("%s.%s.token.acme.invalid", h[:32], h[32:])

	ka, err := keyAuth(c.Key.Public(), token)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	b = sha256.Sum256([]byte(ka))
	h = hex.EncodeToString(b[:])
	sanB := fmt.Sprintf("%s.%s.ka.acme.invalid", h[:32], h[32:])

	cert, err = tlsChallengeCert([]string{sanA, sanB}, opt)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	return cert, sanA, nil
}

// TLSALPN01ChallengeCert creates a certificate for TLS-ALPN-01 challenge response.
// Servers can present the certificate to validate the challenge and prove control
// over a domain name. For more details on TLS-ALPN-01 see
// https://tools.ietf.org/html/draft-shoemaker-acme-tls-alpn-00#section-3
//
// The token argument is a Challenge.Token value.
// If a WithKey option is provided, its private part signs the returned cert,
// and the public part is used to specify the signee.
// If no WithKey option is provided, a new ECDSA key is generated using P-256 curve.
//
// The returned certificate is valid for the next 24 hours and must be presented only when
// the server name in the TLS ClientHello matches the domain, and the special acme-tls/1 ALPN protocol
// has been specified.
func (c *Client) TLSALPN01ChallengeCert(token, domain string, opt ...CertOption) (cert tls.Certificate, err error) {
	ka, err := keyAuth(c.Key.Public(), token)
	if err != nil {
		return tls.Certificate{}, err
	}
	shasum := sha256.Sum256([]byte(ka))
	extValue, err := asn1.Marshal(shasum[:])
	if err != nil {
		return tls.Certificate{}, err
	}
	acmeExtension := pkix.Extension{
		Id:       idPeACMEIdentifier,
		Critical: true,
		Value:    extValue,
	}

	tmpl := defaultTLSChallengeCertTemplate()

	var newOpt []CertOption
	for _, o := range opt {
		switch o := o.(type) {
		case *certOptTemplate:
			t := *(*x509.Certificate)(o) // shallow copy is ok
			tmpl = &t
		default:
			newOpt = append(newOpt, o)
		}
	}
	tmpl.ExtraExtensions = append(tmpl.ExtraExtensions, acmeExtension)
	newOpt = append(newOpt, WithTemplate(tmpl))
	return tlsChallengeCert([]string{domain}, newOpt)
}

// doReg sends all types of registration requests the old way (pre-RFC world).
// The type of request is identified by typ argument, which is a "resource"
// in the ACME spec terms.
//
// A non-nil acct argument indicates whether the intention is to mutate data
// of the Account. Only Contact and Agreement of its fields are used
// in such cases.
func (c *Client) doReg(ctx context.Context, url string, typ string, acct *Account) (*Account, error) {
	req := struct {
		Resource  string   `json:"resource"`
		Contact   []string `json:"contact,omitempty"`
		Agreement string   `json:"agreement,omitempty"`
	}{
		Resource: typ,
	}
	if acct != nil {
		req.Contact = acct.Contact
		req.Agreement = acct.AgreedTerms
	}
	res, err := c.post(ctx, nil, url, req, wantStatus(
		http.StatusOK,       // updates and deletes
		http.StatusCreated,  // new account creation
		http.StatusAccepted, // Let's Encrypt divergent implementation
	))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	var v struct {
		Contact        []string
		Agreement      string
		Authorizations string
		Certificates   string
	}
	if err := json.NewDecoder(res.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("acme: invalid response: %v", err)
	}
	var tos string
	if v := linkHeader(res.Header, "terms-of-service"); len(v) > 0 {
		tos = v[0]
	}
	var authz string
	if v := linkHeader(res.Header, "next"); len(v) > 0 {
		authz = v[0]
	}
	return &Account{
		URI:            res.Header.Get("Location"),
		Contact:        v.Contact,
		AgreedTerms:    v.Agreement,
		CurrentTerms:   tos,
		Authz:          authz,
		Authorizations: v.Authorizations,
		Certificates:   v.Certificates,
	}, nil
}

// popNonce returns a nonce value previously stored with c.addNonce
// or fetches a fresh one from c.dir.NonceURL.
// If NonceURL is empty, it first tries c.directoryURL() and, failing that,
// the provided url.
func (c *Client) popNonce(ctx context.Context, url string) (string, error) {
	c.noncesMu.Lock()
	defer c.noncesMu.Unlock()
	if len(c.nonces) == 0 {
		if c.dir != nil && c.dir.NonceURL != "" {
			return c.fetchNonce(ctx, c.dir.NonceURL)
		}
		dirURL := c.directoryURL()
		v, err := c.fetchNonce(ctx, dirURL)
		if err != nil && url != dirURL {
			v, err = c.fetchNonce(ctx, url)
		}
		return v, err
	}
	var nonce string
	for nonce = range c.nonces {
		delete(c.nonces, nonce)
		break
	}
	return nonce, nil
}

// clearNonces clears any stored nonces
func (c *Client) clearNonces() {
	c.noncesMu.Lock()
	defer c.noncesMu.Unlock()
	c.nonces = make(map[string]struct{})
}

// addNonce stores a nonce value found in h (if any) for future use.
func (c *Client) addNonce(h http.Header) {
	v := nonceFromHeader(h)
	if v == "" {
		return
	}
	c.noncesMu.Lock()
	defer c.noncesMu.Unlock()
	if len(c.nonces) >= maxNonces {
		return
	}
	if c.nonces == nil {
		c.nonces = make(map[string]struct{})
	}
	c.nonces[v] = struct{}{}
}

func (c *Client) fetchNonce(ctx context.Context, url string) (string, error) {
	r, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return "", err
	}
	resp, err := c.doNoRetry(ctx, r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	nonce := nonceFromHeader(resp.Header)
	if nonce == "" {
		if resp.StatusCode > 299 {
			return "", responseError(resp)
		}
		return "", errors.New("acme: nonce not found")
	}
	return nonce, nil
}

func nonceFromHeader(h http.Header) string {
	return h.Get("Replay-Nonce")
}

func (c *Client) responseCert(ctx context.Context, res *http.Response, bundle bool) ([][]byte, error) {
	b, err := ioutil.ReadAll(io.LimitReader(res.Body, maxCertSize+1))
	if err != nil {
		return nil, fmt.Errorf("acme: response stream: %v", err)
	}
	if len(b) > maxCertSize {
		return nil, errors.New("acme: certificate is too big")
	}
	cert := [][]byte{b}
	if !bundle {
		return cert, nil
	}

	// Append CA chain cert(s).
	// At least one is required according to the spec:
	// https://tools.ietf.org/html/draft-ietf-acme-acme-03#section-6.3.1
	up := linkHeader(res.Header, "up")
	if len(up) == 0 {
		return nil, errors.New("acme: rel=up link not found")
	}
	if len(up) > maxChainLen {
		return nil, errors.New("acme: rel=up link is too large")
	}
	for _, url := range up {
		cc, err := c.chainCert(ctx, url, 0)
		if err != nil {
			return nil, err
		}
		cert = append(cert, cc...)
	}
	return cert, nil
}

// chainCert fetches CA certificate chain recursively by following "up" links.
// Each recursive call increments the depth by 1, resulting in an error
// if the recursion level reaches maxChainLen.
//
// First chainCert call starts with depth of 0.
func (c *Client) chainCert(ctx context.Context, url string, depth int) ([][]byte, error) {
	if depth >= maxChainLen {
		return nil, errors.New("acme: certificate chain is too deep")
	}

	res, err := c.get(ctx, url, wantStatus(http.StatusOK))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	b, err := ioutil.ReadAll(io.LimitReader(res.Body, maxCertSize+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxCertSize {
		return nil, errors.New("acme: certificate is too big")
	}
	chain := [][]byte{b}

	uplink := linkHeader(res.Header, "up")
	if len(uplink) > maxChainLen {
		return nil, errors.New("acme: certificate chain is too large")
	}
	for _, up := range uplink {
		cc, err := c.chainCert(ctx, up, depth+1)
		if err != nil {
			return nil, err
		}
		chain = append(chain, cc...)
	}

	return chain, nil
}

// linkHeader returns URI-Reference values of all Link headers
// with relation-type rel.
// See https://tools.ietf.org/html/rfc5988#section-5 for details.
func linkHeader(h http.Header, rel string) []string {
	var links []string
	for _, v := range h["Link"] {
		parts := strings.Split(v, ";")
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if !strings.HasPrefix(p, "rel=") {
				continue
			}
			if v := strings.Trim(p[4:], `"`); v == rel {
				links = append(links, strings.Trim(parts[0], "<>"))
			}
		}
	}
	return links
}

// keyAuth generates a key authorization string for a given token.
func keyAuth(pub crypto.PublicKey, token string) (string, error) {
	th, err := JWKThumbprint(pub)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s.%s", token, th), nil
}

// defaultTLSChallengeCertTemplate is a template used to create challenge certs for TLS challenges.
func defaultTLSChallengeCertTemplate() *x509.Certificate {
	return &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(24 * time.Hour),
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
}

// tlsChallengeCert creates a temporary certificate for TLS-SNI challenges
// with the given SANs and auto-generated public/private key pair.
// The Subject Common Name is set to the first SAN to aid debugging.
// To create a cert with a custom key pair, specify WithKey option.
func tlsChallengeCert(san []string, opt []CertOption) (tls.Certificate, error) {
	var key crypto.Signer
	tmpl := defaultTLSChallengeCertTemplate()
	for _, o := range opt {
		switch o := o.(type) {
		case *certOptKey:
			if key != nil {
				return tls.Certificate{}, errors.New("acme: duplicate key option")
			}
			key = o.key
		case *certOptTemplate:
			t := *(*x509.Certificate)(o) // shallow copy is ok
			tmpl = &t
		default:
			// package's fault, if we let this happen:
			panic(fmt.Sprintf("unsupported option type %T", o))
		}
	}
	if key == nil {
		var err error
		if key, err = ecdsa.GenerateKey(elliptic.P256(), rand.Reader); err != nil {
			return tls.Certificate{}, err
		}
	}
	tmpl.DNSNames = san
	if len(san) > 0 {
		tmpl.Subject.CommonName = san[0]
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, key.Public(), key)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.Certificate{
		Certificate: [][]byte{der},
		PrivateKey:  key,
	}, nil
}

// encodePEM returns b encoded as PEM with block of type typ.
func encodePEM(typ string, b []byte) []byte {
	pb := &pem.Block{Type: typ, Bytes: b}
	return pem.EncodeToMemory(pb)
}

// timeNow is useful for testing for fixed current time.
var timeNow = time.Now
