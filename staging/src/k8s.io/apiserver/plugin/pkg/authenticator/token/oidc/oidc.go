/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
oidc implements the authenticator.Token interface using the OpenID Connect protocol.

	config := oidc.Options{
		IssuerURL:     "https://accounts.google.com",
		ClientID:      os.Getenv("GOOGLE_CLIENT_ID"),
		UsernameClaim: "email",
	}
	tokenAuthenticator, err := oidc.New(config)
*/
package oidc

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	oidc "github.com/coreos/go-oidc"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	certutil "k8s.io/client-go/util/cert"
)

var (
	// synchronizeVerifierForTest is a test hook that, if set to true, blocks
	// token ID verifier initialization so that tests always have a verifier to
	// use.
	synchronizeVerifierForTest = false
)

type Options struct {
	// IssuerURL is the URL the provider signs ID Tokens as. This will be the "iss"
	// field of all tokens produced by the provider and is used for configuration
	// discovery.
	//
	// The URL is usually the provider's URL without a path, for example
	// "https://accounts.google.com" or "https://login.salesforce.com".
	//
	// The provider must implement configuration discovery.
	// See: https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderConfig
	IssuerURL string

	// ClientID the JWT must be issued for, the "sub" field. This plugin only trusts a single
	// client to ensure the plugin can be used with public providers.
	//
	// The plugin supports the "authorized party" OpenID Connect claim, which allows
	// specialized providers to issue tokens to a client for a different client.
	// See: https://openid.net/specs/openid-connect-core-1_0.html#IDToken
	ClientID string

	// Path to a PEM encoded root certificate of the provider.
	CAFile string

	// UsernameClaim is the JWT field to use as the user's username.
	UsernameClaim string

	// UsernamePrefix, if specified, causes claims mapping to username to be prefix with
	// the provided value. A value "oidc:" would result in usernames like "oidc:john".
	UsernamePrefix string

	// GroupsClaim, if specified, causes the OIDCAuthenticator to try to populate the user's
	// groups with an ID Token field. If the GrouppClaim field is present in an ID Token the value
	// must be a string or list of strings.
	GroupsClaim string

	// GroupsPrefix, if specified, causes claims mapping to group names to be prefixed with the
	// value. A value "oidc:" would result in groups like "oidc:engineering" and "oidc:marketing".
	GroupsPrefix string

	// SupportedSigningAlgs sets the accepted set of JOSE signing algorithms that
	// can be used by the provider to sign tokens.
	//
	// https://tools.ietf.org/html/rfc7518#section-3.1
	//
	// This value defaults to RS256, the value recommended by the OpenID Connect
	// spec:
	//
	// https://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation
	SupportedSigningAlgs []string

	// RequiredClaims, if specified, causes the OIDCAuthenticator to verify that all the
	// required claims key value pairs are present in the ID Token.
	RequiredClaims map[string]string

	// now is used for testing. It defaults to time.Now.
	now func() time.Time

	// IssuersPerClaim supports distributed claims.
	//
	// See for details:
	// http://openid.net/specs/openid-connect-core-1_0.html#AggregatedDistributedClaims
	//
	// Lists, for each claim, the list of issuer URLs that are
	// trusted to provide the respective distributed claim.  Only distributed
	// claims with issuers as specified below will be resolved.  This is done
	// to reduce the chance that an authentication request goes to an address
	// that we don't trust to issue distributed claims.
	//
	// Multiple issuers are supported so that it would be possible to rotate
	// the issuers over time (with sufficient advance notice) without
	// disrupting cluster operations, while it is expected that more than one
	// issuer per claim is present only if the distributed claims source change
	// is imminent.
	//
	// Example:
	// "groups" -> []string{"http://example1.com/foo","http://example2.com/bar"}
	IssuersPerClaim map[string][]string
}

// initVerifier creates a new ID token verifier for the given configuration and issuer URL.  On success, calls setVerifier with the
// resulting verifier.  Returns true in case of success, or non-nil error in case of an irrecoverable error.
func initVerifier(ctx context.Context, config *oidc.Config, iss string) (*oidc.IDTokenVerifier, error) {
	provider, err := oidc.NewProvider(ctx, iss)
	if err != nil {
		return nil, fmt.Errorf("init verifier failed: %v", err)
	}
	return provider.Verifier(config), nil
}

// asyncIDTokenVerifier is an ID token verifier that allows async initialization
// of the issuer check.  Must be passed by reference as it wraps sync.Mutex.
type asyncIDTokenVerifier struct {
	m sync.Mutex

	// v is the ID token verifier initialized asynchronously.  It remains nil
	// up until it is eventually initialized.
	// Guarded by m
	v *oidc.IDTokenVerifier
}

// newAsyncIDTokenVerifier creates a new asynchronous token verifier.  The
// verifier is available immediately, but may remain uninitialized for some time
// after creation.
// TODO: newAuthenticator could use this as well.
func newAsyncIDTokenVerifier(ctx context.Context, c *oidc.Config, iss string) *asyncIDTokenVerifier {
	t := &asyncIDTokenVerifier{}

	// Polls indefinitely in an attemt to initialize the distributed claims
	// verifier, or until context canceled.
	sync := make(chan struct{})
	initFn := func() (done bool, err error) {
		glog.V(10).Infof("oidc authenticator: attempting init: iss=%v", iss)
		v, err := initVerifier(ctx, c, iss)
		if err != nil {
			glog.Errorf("oidc authenticator: async token verifier for issuer: %q: %v", iss, err)
			return false, nil
		}
		t.m.Lock()
		defer t.m.Unlock()
		t.v = v
		close(sync)
		return true, nil

	}
	go func() {
		if done, _ := initFn(); !done {
			go wait.PollUntil(time.Second*10, initFn, ctx.Done())
		}
	}()

	// Let tests wait until the verifier is initialized.
	if synchronizeVerifierForTest {
		<-sync
	}
	return t
}

// verifier returns the underlying ID token verifier, or nil if one is not yet initialized.
func (a *asyncIDTokenVerifier) verifier() *oidc.IDTokenVerifier {
	a.m.Lock()
	defer a.m.Unlock()
	return a.v
}

type Authenticator struct {
	issuerURL string

	usernameClaim  string
	usernamePrefix string
	groupsClaim    string
	groupsPrefix   string
	requiredClaims map[string]string

	// Contains an *oidc.IDTokenVerifier. Do not access directly use the
	// idTokenVerifier method.
	verifier atomic.Value

	cancel context.CancelFunc

	claimsProcessor *claimResolver
}

func (a *Authenticator) setVerifier(v *oidc.IDTokenVerifier) {
	a.verifier.Store(v)
}

func (a *Authenticator) idTokenVerifier() (*oidc.IDTokenVerifier, bool) {
	if v := a.verifier.Load(); v != nil {
		return v.(*oidc.IDTokenVerifier), true
	}
	return nil, false
}

func (a *Authenticator) Close() {
	a.cancel()
}

func New(opts Options) (*Authenticator, error) {
	return newAuthenticator(opts, func(ctx context.Context, a *Authenticator, config *oidc.Config) {
		// Asynchronously attempt to initialize the authenticator. This enables
		// self-hosted providers, providers that run on top of Kubernetes itself.
		go wait.PollUntil(time.Second*10, func() (done bool, err error) {
			provider, err := oidc.NewProvider(ctx, a.issuerURL)
			if err != nil {
				glog.Errorf("oidc authenticator: initializing plugin: %v", err)
				return false, nil
			}

			verifier := provider.Verifier(config)
			a.setVerifier(verifier)
			return true, nil
		}, ctx.Done())
	})
}

// whitelist of signing algorithms to ensure users don't mistakenly pass something
// goofy.
var allowedSigningAlgs = map[string]bool{
	oidc.RS256: true,
	oidc.RS384: true,
	oidc.RS512: true,
	oidc.ES256: true,
	oidc.ES384: true,
	oidc.ES512: true,
	oidc.PS256: true,
	oidc.PS384: true,
	oidc.PS512: true,
}

func newAuthenticator(opts Options, initVerifier func(ctx context.Context, a *Authenticator, config *oidc.Config)) (*Authenticator, error) {
	url, err := url.Parse(opts.IssuerURL)
	if err != nil {
		return nil, err
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", opts.IssuerURL, url.Scheme)
	}

	if opts.UsernameClaim == "" {
		return nil, errors.New("no username claim provided")
	}

	supportedSigningAlgs := opts.SupportedSigningAlgs
	if len(supportedSigningAlgs) == 0 {
		// RS256 is the default recommended by OpenID Connect and an 'alg' value
		// providers are required to implement.
		supportedSigningAlgs = []string{oidc.RS256}
	}
	for _, alg := range supportedSigningAlgs {
		if !allowedSigningAlgs[alg] {
			return nil, fmt.Errorf("oidc: unsupported signing alg: %q", alg)
		}
	}

	var roots *x509.CertPool
	if opts.CAFile != "" {
		roots, err = certutil.NewPool(opts.CAFile)
		if err != nil {
			return nil, fmt.Errorf("Failed to read the CA file: %v", err)
		}
	} else {
		glog.Info("OIDC: No x509 certificates provided, will use host's root CA set")
	}

	// Copied from http.DefaultTransport.
	tr := net.SetTransportDefaults(&http.Transport{
		// According to golang's doc, if RootCAs is nil,
		// TLS uses the host's root CA set.
		TLSClientConfig: &tls.Config{RootCAs: roots},
	})

	client := &http.Client{Transport: tr, Timeout: 30 * time.Second}

	ctx, cancel := context.WithCancel(context.Background())
	ctx = oidc.ClientContext(ctx, client)

	now := opts.now
	if now == nil {
		now = time.Now
	}

	verifierConfig := &oidc.Config{
		ClientID:             opts.ClientID,
		SupportedSigningAlgs: supportedSigningAlgs,
		Now:                  now,
	}

	authenticator := &Authenticator{
		issuerURL:       opts.IssuerURL,
		usernameClaim:   opts.UsernameClaim,
		usernamePrefix:  opts.UsernamePrefix,
		groupsClaim:     opts.GroupsClaim,
		groupsPrefix:    opts.GroupsPrefix,
		requiredClaims:  opts.RequiredClaims,
		cancel:          cancel,
		claimsProcessor: newClaimResolver(ctx, opts.IssuersPerClaim, tr, verifierConfig),
	}

	initVerifier(ctx, authenticator, verifierConfig)
	return authenticator, nil
}

// untrustedIssuer extracts an untrusted "iss" claim from the given JWT token,
// or returns an error if the token can not be parsed.  Since the JWT is not
// verified, the returned issuer should not be trusted.
func untrustedIssuer(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("malformed token %q", token)
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("error decoding token %q", token)
	}
	claims := struct {
		// WARNING: this JWT is not verified. Do not trust these claims.
		Issuer string `json:"iss"`
	}{}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("while unmarshaling token: %q: %v", err)
	}
	return claims.Issuer, nil
}

func hasCorrectIssuer(iss, tokenData string) bool {
	uiss, err := untrustedIssuer(tokenData)
	if err != nil {
		glog.Errorf("while checking issuer: %v", err)
		return false
	}
	if uiss != iss {
		glog.V(11).Infof("Issuer mismatch: want=%q, got=%q", iss, uiss)
		return false
	}
	return true
}

// endpoint represents an OIDC distributed claims endpoint.
type endpoint struct {
	// URL to use to request the distributed claim.  This URL is expected to be
	// prefixed by one of the known issuer URLs.
	URL string `json:"endpoint,omitempty"`
	// AccessToken is the bearer token to use for access.  If empty, it is
	// not used.  Access token is optional per the OIDC distributed claims
	// specification.
	// See: http://openid.net/specs/openid-connect-core-1_0.html#DistributedExample
	AccessToken string `json:"access_token,omitempty"`
	// JWT is the container for aggregated claims.  Not supported at the moment.
	// See: http://openid.net/specs/openid-connect-core-1_0.html#AggregatedExample
	JWT string `json:"JWT,omitempty"`
}

// claimInfo contains information used to verify a single claim.
type claimInfo struct {
	// name is the claim name, e.g. "groups"
	name string

	// verifierPerIssuer contains, for each issuer, the appropriate verifier to use
	// for this claim.
	verifierPerIssuer map[string]*asyncIDTokenVerifier
}

func (i claimInfo) String() string {
	return fmt.Sprintf("name:%q, verifierPerIssuer:%+v", i.name, i.verifierPerIssuer)
}

// claimResolver expands distributed claims by calling respective claim source
// endpoints.
type claimResolver struct {
	// claimInfos is a map from a claim to verifier map.
	claimInfos map[string]claimInfo

	// A HTTP transport to use for distributed claims
	t *http.Transport
}

// newClaimResolver creates a new resolver for distributed claims.  Each claim
// has a number of allowed issuers, as specified by issuersPerClaim, and only
// those issuers are honored.
// TODO: Support separate client IDs per different issuer/claim.
func newClaimResolver(ctx context.Context, issuersPerClaim map[string][]string, t *http.Transport, config *oidc.Config) *claimResolver {
	cp := &claimResolver{claimInfos: map[string]claimInfo{}, t: t}
	for c, issuers := range issuersPerClaim {
		verifierPerIssuer := map[string]*asyncIDTokenVerifier{}
		for _, iss := range issuers {
			glog.V(10).Infof("adding claim info: %v", iss)
			verifierPerIssuer[iss] = newAsyncIDTokenVerifier(ctx, config, iss)
		}
		ci := claimInfo{c, verifierPerIssuer}
		cp.claimInfos[c] = ci
		glog.V(10).Infof("adding claim info: %+v", ci)
	}
	return cp
}

// Verifier returns either the verifier for the specified issuer, or error.
func (p *claimResolver) Verifier(claim, iss string) (*oidc.IDTokenVerifier, error) {
	ci, ok := p.claimInfos[claim]
	if !ok {
		return nil, fmt.Errorf("no claim info for claim: %q, issuer: %q", claim, iss)
	}
	av := ci.verifierPerIssuer[iss]
	if av == nil {
		return nil, fmt.Errorf("no verifier for claim: %q, issuer: %q", claim, iss)
	}
	v := av.verifier()
	if v == nil {
		return nil, fmt.Errorf("verifier not initialized for claim: %q, issuer: %q", claim, iss)
	}
	glog.V(10).Infof("Got verifier for: claim: %q, iss: %q: %+v", claim, iss, v)
	return v, nil
}

// expand extracts the distributed claims from claim names and claim sources.
// The extracted claim value is pulled up into the supplied claims.
//
// Distributed claims are of the form as seen below, and are defined in the
// OIDC Connect Core 1.0, section 5.6.2.
// See: https://openid.net/specs/openid-connect-core-1_0.html#AggregatedDistributedClaims
//
// {
//   ... (other normal claims)...
//   "_claim_names": {
//     "groups": "src1"
//   },
//   "_claim_sources": {
//     "src1": {
//       "endpoint": "https://www.example.com",
//       "access_token": "f005ba11"
//     },
//   },
// }
func (r *claimResolver) expand(c claims) error {
	const (
		// The claim containing a map of endpoint references per claim.
		// OIDC Connect Core 1.0, section 5.6.2.
		claimNamesKey = "_claim_names"
		// The claim containing endpoint specifications.
		// OIDC Connect Core 1.0, section 5.6.2.
		claimSourcesKey = "_claim_sources"
	)

	if glog.V(11) {
		glog.Infof("initial claim set: %v", c)
	}
	defer func() {
		if glog.V(10) {
			glog.Infof("final claim set: %v", c)
		}
	}()
	names, ok := c[claimNamesKey]
	if !ok {
		// No _claim_names, no keys to look up.
		return nil
	}

	claimToSource := map[string]string{}
	if err := json.Unmarshal([]byte(names), &claimToSource); err != nil {
		return fmt.Errorf("oidc: error parsing distributed claim names: %v", err)
	}

	var rawSources json.RawMessage
	rawSources, ok = c[claimSourcesKey]
	if !ok {
		// Having _claim_names claim,  but no _claim_sources is not an expected
		// state.
		return fmt.Errorf("oidc: no claim sources")
	}

	var sources claims
	if err := json.Unmarshal([]byte(rawSources), &sources); err != nil {
		// The claims sources claim is malformed, this is not an expected state.
		return fmt.Errorf("oidc: could not parse claim sources")
	}

	// Build a map of claims to endpoint, resolving the indirection between
	// _claim_names and _claim_sources.
	cep := map[string]endpoint{}
	for claim, source := range claimToSource {
		if _, ok := c[claim]; ok {
			// Skip distributed claim if a normal claim is already present.
			continue
		}
		rawEndpoint, ok := sources[source]
		if !ok {
			// We expect that if a claim is named, that the corresponding source exists, too.
			return fmt.Errorf("missing source for claim: %v", claim)
		}
		var e endpoint
		if err := json.Unmarshal([]byte(rawEndpoint), &e); err != nil {
			return fmt.Errorf("could not unmarshal claim: %v", err)
		}
		if e.URL == "" {
			// Ignoring aggregate claims for now.
			continue
		}
		cep[claim] = e
	}
	return r.resolve(cep, c)
}

// resolve requests distributed claims from all endpoints passed in,
// and inserts the lookup results into allClaims.
func (p *claimResolver) resolve(endpoints map[string]endpoint, allClaims claims) error {
	for claim, endpoint := range endpoints {
		// This could be parallelized if needed.
		// The endpoint MUST return a JWT per
		// http://openid.net/specs/openid-connect-core-1_0.html#AggregatedDistributedClaims
		// TODO: cache resolved claims.
		// TODO: support aggregated claims.
		jwt, err := getClaimJWT(p.t, endpoint.URL, endpoint.AccessToken)
		if err != nil {
			return fmt.Errorf("while getting distributed claim %q: %v", claim, err)
		}
		untrustedIss, err := untrustedIssuer(jwt)
		if err != nil {
			glog.Errorf("while getting issuer: %q: %v", claim, err)
			continue
		}
		v, err := p.Verifier(claim, untrustedIss)
		if err != nil {
			glog.Errorf("verifier: %v", err)
			continue
		}
		t, err := v.Verify(context.Background(), jwt)
		if err != nil {
			return fmt.Errorf("verify distributed claim token: %v", err)
		}
		var distClaims claims
		if err := t.Claims(&distClaims); err != nil {
			return fmt.Errorf("could not parse distributed claims for claim %v: %v", claim, err)
		}
		value, ok := distClaims[claim]
		if !ok {
			return fmt.Errorf("could not find distributed claim: %v", claim)
		}
		allClaims[claim] = value
	}
	return nil
}

func (a *Authenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	if !hasCorrectIssuer(a.issuerURL, token) {
		glog.V(10).Infof("Issuer mismatch at entry.")
		return nil, false, nil
	}

	verifier, ok := a.idTokenVerifier()
	if !ok {
		return nil, false, fmt.Errorf("oidc: authenticator not initialized")
	}

	ctx := context.Background()
	idToken, err := verifier.Verify(ctx, token)
	if err != nil {
		return nil, false, fmt.Errorf("oidc: verify token: %v", err)
	}

	var c claims
	if err := idToken.Claims(&c); err != nil {
		return nil, false, fmt.Errorf("oidc: parse claims: %v", err)
	}
	if err := a.claimsProcessor.expand(c); err != nil {
		return nil, false, fmt.Errorf("oidc: could not expand distributed claims: %v", err)
	}

	var username string
	if err := c.unmarshalClaim(a.usernameClaim, &username); err != nil {
		return nil, false, fmt.Errorf("oidc: parse username claims %q: %v", a.usernameClaim, err)
	}

	if a.usernameClaim == "email" {
		// If the email_verified claim is present, ensure the email is valid.
		// https://openid.net/specs/openid-connect-core-1_0.html#StandardClaims
		if hasEmailVerified := c.hasClaim("email_verified"); hasEmailVerified {
			var emailVerified bool
			if err := c.unmarshalClaim("email_verified", &emailVerified); err != nil {
				return nil, false, fmt.Errorf("oidc: parse 'email_verified' claim: %v", err)
			}

			// If the email_verified claim is present we have to verify it is set to `true`.
			if !emailVerified {
				return nil, false, fmt.Errorf("oidc: email not verified")
			}
		}
	}

	if a.usernamePrefix != "" {
		username = a.usernamePrefix + username
	}

	info := &user.DefaultInfo{Name: username}
	if a.groupsClaim != "" {
		if _, ok := c[a.groupsClaim]; ok {
			// Some admins want to use string claims like "role" as the group value.
			// Allow the group claim to be a single string instead of an array.
			//
			// See: https://github.com/kubernetes/kubernetes/issues/33290
			var groups stringOrArray
			if err := c.unmarshalClaim(a.groupsClaim, &groups); err != nil {
				return nil, false, fmt.Errorf("oidc: parse groups claim %q: %v", a.groupsClaim, err)
			}
			info.Groups = []string(groups)
		}
	}

	if a.groupsPrefix != "" {
		for i, group := range info.Groups {
			info.Groups[i] = a.groupsPrefix + group
		}
	}

	// check to ensure all required claims are present in the ID token and have matching values.
	for claim, value := range a.requiredClaims {
		if !c.hasClaim(claim) {
			return nil, false, fmt.Errorf("oidc: required claim %s not present in ID token", claim)
		}

		// NOTE: Only string values are supported as valid required claim values.
		var claimValue string
		if err := c.unmarshalClaim(claim, &claimValue); err != nil {
			return nil, false, fmt.Errorf("oidc: parse claim %s: %v", claim, err)
		}
		if claimValue != value {
			return nil, false, fmt.Errorf("oidc: required claim %s value does not match. Got = %s, want = %s", claim, claimValue, value)
		}
	}

	return info, true, nil
}

// getClaimJWT gets a distributed claim JWT from url, using the supplied access
// token as bearer token.  If the access token is "", the authorization header
// will not be set.
// TODO: Allow passing in JSON hints to the IDP.
func getClaimJWT(t *http.Transport, url, accessToken string) (string, error) {
	client := &http.Client{Transport: t, Timeout: 30 * time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// TODO: Allow passing request body with configurable information.
	req, _ := http.NewRequest("GET", url, nil)
	if accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %v", accessToken))
	}
	req.WithContext(ctx)
	response, err := client.Do(req)
	if err != nil {
		return "", err
	}
	responseBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", fmt.Errorf("could not decode distributed claim response")
	}
	return string(responseBytes), nil
}

type stringOrArray []string

func (s *stringOrArray) UnmarshalJSON(b []byte) error {
	var a []string
	if err := json.Unmarshal(b, &a); err == nil {
		*s = a
		return nil
	}
	var str string
	if err := json.Unmarshal(b, &str); err != nil {
		return err
	}
	*s = []string{str}
	return nil
}

type claims map[string]json.RawMessage

func (c claims) unmarshalClaim(name string, v interface{}) error {
	val, ok := c[name]
	if !ok {
		return fmt.Errorf("claim not present")
	}
	return json.Unmarshal([]byte(val), v)
}

func (c claims) hasClaim(name string) bool {
	if _, ok := c[name]; !ok {
		return false
	}
	return true
}

func (c claims) String() string {
	r := map[string]string{}
	for k, v := range c {
		r[k] = string(v)
	}
	return fmt.Sprintf("%+v", r)
}
