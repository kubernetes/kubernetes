// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
Package confidential provides a client for authentication of "confidential" applications.
A "confidential" application is defined as an app that run on servers. They are considered
difficult to access and for that reason capable of keeping an application secret.
Confidential clients can hold configuration-time secrets.
*/
package confidential

import (
	"context"
	"crypto"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"net/url"

	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/cache"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/base"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/exported"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/accesstokens"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/authority"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/options"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/shared"
)

/*
Design note:

confidential.Client uses base.Client as an embedded type. base.Client statically assigns its attributes
during creation. As it doesn't have any pointers in it, anything borrowed from it, such as
Base.AuthParams is a copy that is free to be manipulated here.

Duplicate Calls shared between public.Client and this package:
There is some duplicate call options provided here that are the same as in public.Client . This
is a design choices. Go proverb(https://www.youtube.com/watch?v=PAAkCSZUG1c&t=9m28s):
"a little copying is better than a little dependency". Yes, we could have another package with
shared options (fail).  That divides like 2 options from all others which makes the user look
through more docs.  We can have all clients in one package, but I think separate packages
here makes for better naming (public.Client vs client.PublicClient).  So I chose a little
duplication.

.Net People, Take note on X509:
This uses x509.Certificates and private keys. x509 does not store private keys. .Net
has some x509.Certificate2 thing that has private keys, but that is just some bullcrap that .Net
added, it doesn't exist in real life.  Seriously, "x509.Certificate2", bahahahaha.  As such I've
put a PEM decoder into here.
*/

// TODO(msal): This should have example code for each method on client using Go's example doc framework.
// base usage details should be include in the package documentation.

// AuthResult contains the results of one token acquisition operation.
// For details see https://aka.ms/msal-net-authenticationresult
type AuthResult = base.AuthResult

type Account = shared.Account

// CertFromPEM converts a PEM file (.pem or .key) for use with NewCredFromCert(). The file
// must contain the public certificate and the private key. If a PEM block is encrypted and
// password is not an empty string, it attempts to decrypt the PEM blocks using the password.
// Multiple certs are due to certificate chaining for use cases like TLS that sign from root to leaf.
func CertFromPEM(pemData []byte, password string) ([]*x509.Certificate, crypto.PrivateKey, error) {
	var certs []*x509.Certificate
	var priv crypto.PrivateKey
	for {
		block, rest := pem.Decode(pemData)
		if block == nil {
			break
		}

		//nolint:staticcheck // x509.IsEncryptedPEMBlock and x509.DecryptPEMBlock are deprecated. They are used here only to support a usecase.
		if x509.IsEncryptedPEMBlock(block) {
			b, err := x509.DecryptPEMBlock(block, []byte(password))
			if err != nil {
				return nil, nil, fmt.Errorf("could not decrypt encrypted PEM block: %v", err)
			}
			block, _ = pem.Decode(b)
			if block == nil {
				return nil, nil, fmt.Errorf("encounter encrypted PEM block that did not decode")
			}
		}

		switch block.Type {
		case "CERTIFICATE":
			cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, nil, fmt.Errorf("block labelled 'CERTIFICATE' could not be parsed by x509: %v", err)
			}
			certs = append(certs, cert)
		case "PRIVATE KEY":
			if priv != nil {
				return nil, nil, errors.New("found multiple private key blocks")
			}

			var err error
			priv, err = x509.ParsePKCS8PrivateKey(block.Bytes)
			if err != nil {
				return nil, nil, fmt.Errorf("could not decode private key: %v", err)
			}
		case "RSA PRIVATE KEY":
			if priv != nil {
				return nil, nil, errors.New("found multiple private key blocks")
			}
			var err error
			priv, err = x509.ParsePKCS1PrivateKey(block.Bytes)
			if err != nil {
				return nil, nil, fmt.Errorf("could not decode private key: %v", err)
			}
		}
		pemData = rest
	}

	if len(certs) == 0 {
		return nil, nil, fmt.Errorf("no certificates found")
	}

	if priv == nil {
		return nil, nil, fmt.Errorf("no private key found")
	}

	return certs, priv, nil
}

// AssertionRequestOptions has required information for client assertion claims
type AssertionRequestOptions = exported.AssertionRequestOptions

// Credential represents the credential used in confidential client flows.
type Credential struct {
	secret string

	cert *x509.Certificate
	key  crypto.PrivateKey
	x5c  []string

	assertionCallback func(context.Context, AssertionRequestOptions) (string, error)

	tokenProvider func(context.Context, TokenProviderParameters) (TokenProviderResult, error)
}

// toInternal returns the accesstokens.Credential that is used internally. The current structure of the
// code requires that client.go, requests.go and confidential.go share a credential type without
// having import recursion. That requires the type used between is in a shared package. Therefore
// we have this.
func (c Credential) toInternal() (*accesstokens.Credential, error) {
	if c.secret != "" {
		return &accesstokens.Credential{Secret: c.secret}, nil
	}
	if c.cert != nil {
		if c.key == nil {
			return nil, errors.New("missing private key for certificate")
		}
		return &accesstokens.Credential{Cert: c.cert, Key: c.key, X5c: c.x5c}, nil
	}
	if c.key != nil {
		return nil, errors.New("missing certificate for private key")
	}
	if c.assertionCallback != nil {
		return &accesstokens.Credential{AssertionCallback: c.assertionCallback}, nil
	}
	if c.tokenProvider != nil {
		return &accesstokens.Credential{TokenProvider: c.tokenProvider}, nil
	}
	return nil, errors.New("invalid credential")
}

// NewCredFromSecret creates a Credential from a secret.
func NewCredFromSecret(secret string) (Credential, error) {
	if secret == "" {
		return Credential{}, errors.New("secret can't be empty string")
	}
	return Credential{secret: secret}, nil
}

// NewCredFromAssertion creates a Credential from a signed assertion.
//
// Deprecated: a Credential created by this function can't refresh the
// assertion when it expires. Use NewCredFromAssertionCallback instead.
func NewCredFromAssertion(assertion string) (Credential, error) {
	if assertion == "" {
		return Credential{}, errors.New("assertion can't be empty string")
	}
	return NewCredFromAssertionCallback(func(context.Context, AssertionRequestOptions) (string, error) { return assertion, nil }), nil
}

// NewCredFromAssertionCallback creates a Credential that invokes a callback to get assertions
// authenticating the application. The callback must be thread safe.
func NewCredFromAssertionCallback(callback func(context.Context, AssertionRequestOptions) (string, error)) Credential {
	return Credential{assertionCallback: callback}
}

// NewCredFromCert creates a Credential from an x509.Certificate and an RSA private key.
// CertFromPEM() can be used to get these values from a PEM file.
func NewCredFromCert(cert *x509.Certificate, key crypto.PrivateKey) Credential {
	cred, _ := NewCredFromCertChain([]*x509.Certificate{cert}, key)
	return cred
}

// NewCredFromCertChain creates a Credential from a chain of x509.Certificates and an RSA private key
// as returned by CertFromPEM().
func NewCredFromCertChain(certs []*x509.Certificate, key crypto.PrivateKey) (Credential, error) {
	cred := Credential{key: key}
	k, ok := key.(*rsa.PrivateKey)
	if !ok {
		return cred, errors.New("key must be an RSA key")
	}
	for _, cert := range certs {
		if cert == nil {
			// not returning an error here because certs may still contain a sufficient cert/key pair
			continue
		}
		certKey, ok := cert.PublicKey.(*rsa.PublicKey)
		if ok && k.E == certKey.E && k.N.Cmp(certKey.N) == 0 {
			// We know this is the signing cert because its public key matches the given private key.
			// This cert must be first in x5c.
			cred.cert = cert
			cred.x5c = append([]string{base64.StdEncoding.EncodeToString(cert.Raw)}, cred.x5c...)
		} else {
			cred.x5c = append(cred.x5c, base64.StdEncoding.EncodeToString(cert.Raw))
		}
	}
	if cred.cert == nil {
		return cred, errors.New("key doesn't match any certificate")
	}
	return cred, nil
}

// TokenProviderParameters is the authentication parameters passed to token providers
type TokenProviderParameters = exported.TokenProviderParameters

// TokenProviderResult is the authentication result returned by custom token providers
type TokenProviderResult = exported.TokenProviderResult

// NewCredFromTokenProvider creates a Credential from a function that provides access tokens. The function
// must be concurrency safe. This is intended only to allow the Azure SDK to cache MSI tokens. It isn't
// useful to applications in general because the token provider must implement all authentication logic.
func NewCredFromTokenProvider(provider func(context.Context, TokenProviderParameters) (TokenProviderResult, error)) Credential {
	return Credential{tokenProvider: provider}
}

// AutoDetectRegion instructs MSAL Go to auto detect region for Azure regional token service.
func AutoDetectRegion() string {
	return "TryAutoDetect"
}

// Client is a representation of authentication client for confidential applications as defined in the
// package doc. A new Client should be created PER SERVICE USER.
// For more information, visit https://docs.microsoft.com/azure/active-directory/develop/msal-client-applications
type Client struct {
	base base.Client

	cred *accesstokens.Credential

	// userID is some unique identifier for a user. It actually isn't used by us at all, it
	// simply acts as another hint that a confidential.Client is for a single user.
	userID string
}

// Options are optional settings for New(). These options are set using various functions
// returning Option calls.
type Options struct {
	// Accessor controls cache persistence.
	// By default there is no cache persistence. This can be set using the WithAccessor() option.
	Accessor cache.ExportReplace

	// The host of the Azure Active Directory authority.
	// The default is https://login.microsoftonline.com/common. This can be changed using the
	// WithAuthority() option.
	Authority string

	// The HTTP client used for making requests.
	// It defaults to a shared http.Client.
	HTTPClient ops.HTTPClient

	// SendX5C specifies if x5c claim(public key of the certificate) should be sent to STS.
	SendX5C bool

	// Instructs MSAL Go to use an Azure regional token service with sepcified AzureRegion.
	AzureRegion string

	capabilities []string

	disableInstanceDiscovery bool
}

func (o Options) validate() error {
	u, err := url.Parse(o.Authority)
	if err != nil {
		return fmt.Errorf("the Authority(%s) does not parse as a valid URL", o.Authority)
	}
	if u.Scheme != "https" {
		return fmt.Errorf("the Authority(%s) does not appear to use https", o.Authority)
	}
	return nil
}

// Option is an optional argument to New().
type Option func(o *Options)

// WithAuthority allows you to provide a custom authority for use in the client.
func WithAuthority(authority string) Option {
	return func(o *Options) {
		o.Authority = authority
	}
}

// WithAccessor provides a cache accessor that will read and write to some externally managed cache
// that may or may not be shared with other applications.
func WithAccessor(accessor cache.ExportReplace) Option {
	return func(o *Options) {
		o.Accessor = accessor
	}
}

// WithClientCapabilities allows configuring one or more client capabilities such as "CP1"
func WithClientCapabilities(capabilities []string) Option {
	return func(o *Options) {
		// there's no danger of sharing the slice's underlying memory with the application because
		// this slice is simply passed to base.WithClientCapabilities, which copies its data
		o.capabilities = capabilities
	}
}

// WithHTTPClient allows for a custom HTTP client to be set.
func WithHTTPClient(httpClient ops.HTTPClient) Option {
	return func(o *Options) {
		o.HTTPClient = httpClient
	}
}

// WithX5C specifies if x5c claim(public key of the certificate) should be sent to STS to enable Subject Name Issuer Authentication.
func WithX5C() Option {
	return func(o *Options) {
		o.SendX5C = true
	}
}

// WithInstanceDiscovery set to false to disable authority validation (to support private cloud scenarios)
func WithInstanceDiscovery(enabled bool) Option {
	return func(o *Options) {
		o.disableInstanceDiscovery = !enabled
	}
}

// WithAzureRegion sets the region(preferred) or Confidential.AutoDetectRegion() for auto detecting region.
// Region names as per https://azure.microsoft.com/en-ca/global-infrastructure/geographies/.
// See https://aka.ms/region-map for more details on region names.
// The region value should be short region name for the region where the service is deployed.
// For example "centralus" is short name for region Central US.
// Not all auth flows can use the regional token service.
// Service To Service (client credential flow) tokens can be obtained from the regional service.
// Requires configuration at the tenant level.
// Auto-detection works on a limited number of Azure artifacts (VMs, Azure functions).
// If auto-detection fails, the non-regional endpoint will be used.
// If an invalid region name is provided, the non-regional endpoint MIGHT be used or the token request MIGHT fail.
func WithAzureRegion(val string) Option {
	return func(o *Options) {
		o.AzureRegion = val
	}
}

// New is the constructor for Client. userID is the unique identifier of the user this client
// will store credentials for (a Client is per user). clientID is the Azure clientID and cred is
// the type of credential to use.
func New(clientID string, cred Credential, options ...Option) (Client, error) {
	internalCred, err := cred.toInternal()
	if err != nil {
		return Client{}, err
	}

	opts := Options{
		Authority:  base.AuthorityPublicCloud,
		HTTPClient: shared.DefaultClient,
	}

	for _, o := range options {
		o(&opts)
	}
	if err := opts.validate(); err != nil {
		return Client{}, err
	}

	baseOpts := []base.Option{
		base.WithCacheAccessor(opts.Accessor),
		base.WithClientCapabilities(opts.capabilities),
		base.WithRegionDetection(opts.AzureRegion),
		base.WithX5C(opts.SendX5C),
		base.WithInstanceDiscovery(!opts.disableInstanceDiscovery),
	}
	if cred.tokenProvider != nil {
		// The caller will handle all details of authentication, using Client only as a token cache.
		baseOpts = append(baseOpts, base.WithInstanceDiscovery(false))
	}
	base, err := base.New(clientID, opts.Authority, oauth.New(opts.HTTPClient), baseOpts...)
	if err != nil {
		return Client{}, err
	}
	base.AuthParams.IsConfidentialClient = true

	return Client{base: base, cred: internalCred}, nil
}

// UserID is the unique user identifier this client if for.
func (cca Client) UserID() string {
	return cca.userID
}

// authCodeURLOptions contains options for AuthCodeURL
type authCodeURLOptions struct {
	claims, loginHint, tenantID, domainHint string
}

// AuthCodeURLOption is implemented by options for AuthCodeURL
type AuthCodeURLOption interface {
	authCodeURLOption()
}

// AuthCodeURL creates a URL used to acquire an authorization code. Users need to call CreateAuthorizationCodeURLParameters and pass it in.
//
// Options: [WithClaims], [WithDomainHint], [WithLoginHint], [WithTenantID]
func (cca Client) AuthCodeURL(ctx context.Context, clientID, redirectURI string, scopes []string, opts ...AuthCodeURLOption) (string, error) {
	o := authCodeURLOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return "", err
	}
	ap, err := cca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return "", err
	}
	ap.Claims = o.claims
	ap.LoginHint = o.loginHint
	ap.DomainHint = o.domainHint
	return cca.base.AuthCodeURL(ctx, clientID, redirectURI, scopes, ap)
}

// WithLoginHint pre-populates the login prompt with a username.
func WithLoginHint(username string) interface {
	AuthCodeURLOption
	options.CallOption
} {
	return struct {
		AuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *authCodeURLOptions:
					t.loginHint = username
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// WithDomainHint adds the IdP domain as domain_hint query parameter in the auth url.
func WithDomainHint(domain string) interface {
	AuthCodeURLOption
	options.CallOption
} {
	return struct {
		AuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *authCodeURLOptions:
					t.domainHint = domain
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// WithClaims sets additional claims to request for the token, such as those required by conditional access policies.
// Use this option when Azure AD returned a claims challenge for a prior request. The argument must be decoded.
// This option is valid for any token acquisition method.
func WithClaims(claims string) interface {
	AcquireByAuthCodeOption
	AcquireByCredentialOption
	AcquireOnBehalfOfOption
	AcquireSilentOption
	AuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireByAuthCodeOption
		AcquireByCredentialOption
		AcquireOnBehalfOfOption
		AcquireSilentOption
		AuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenByAuthCodeOptions:
					t.claims = claims
				case *acquireTokenByCredentialOptions:
					t.claims = claims
				case *acquireTokenOnBehalfOfOptions:
					t.claims = claims
				case *AcquireTokenSilentOptions:
					t.claims = claims
				case *authCodeURLOptions:
					t.claims = claims
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// WithTenantID specifies a tenant for a single authentication. It may be different than the tenant set in [New] by [WithAuthority].
// This option is valid for any token acquisition method.
func WithTenantID(tenantID string) interface {
	AcquireByAuthCodeOption
	AcquireByCredentialOption
	AcquireOnBehalfOfOption
	AcquireSilentOption
	AuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireByAuthCodeOption
		AcquireByCredentialOption
		AcquireOnBehalfOfOption
		AcquireSilentOption
		AuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenByAuthCodeOptions:
					t.tenantID = tenantID
				case *acquireTokenByCredentialOptions:
					t.tenantID = tenantID
				case *acquireTokenOnBehalfOfOptions:
					t.tenantID = tenantID
				case *AcquireTokenSilentOptions:
					t.tenantID = tenantID
				case *authCodeURLOptions:
					t.tenantID = tenantID
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// AcquireTokenSilentOptions are all the optional settings to an AcquireTokenSilent() call.
// These are set by using various AcquireTokenSilentOption functions.
type AcquireTokenSilentOptions struct {
	// Account represents the account to use. To set, use the WithSilentAccount() option.
	Account Account

	claims, tenantID string
}

// AcquireSilentOption is implemented by options for AcquireTokenSilent
type AcquireSilentOption interface {
	acquireSilentOption()
}

// AcquireTokenSilentOption changes options inside AcquireTokenSilentOptions used in .AcquireTokenSilent().
type AcquireTokenSilentOption func(a *AcquireTokenSilentOptions)

func (AcquireTokenSilentOption) acquireSilentOption() {}

// WithSilentAccount uses the passed account during an AcquireTokenSilent() call.
func WithSilentAccount(account Account) interface {
	AcquireSilentOption
	options.CallOption
} {
	return struct {
		AcquireSilentOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenSilentOptions:
					t.Account = account
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// AcquireTokenSilent acquires a token from either the cache or using a refresh token.
//
// Options: [WithClaims], [WithSilentAccount], [WithTenantID]
func (cca Client) AcquireTokenSilent(ctx context.Context, scopes []string, opts ...AcquireSilentOption) (AuthResult, error) {
	o := AcquireTokenSilentOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}

	if o.claims != "" {
		return AuthResult{}, errors.New("call another AcquireToken method to request a new token having these claims")
	}

	silentParameters := base.AcquireTokenSilentParameters{
		Scopes:      scopes,
		Account:     o.Account,
		RequestType: accesstokens.ATConfidential,
		Credential:  cca.cred,
		IsAppCache:  o.Account.IsZero(),
		TenantID:    o.tenantID,
	}

	return cca.base.AcquireTokenSilent(ctx, silentParameters)
}

// AcquireTokenByAuthCodeOptions contains the optional parameters used to acquire an access token using the authorization code flow.
type AcquireTokenByAuthCodeOptions struct {
	Challenge string

	claims, tenantID string
}

// AcquireByAuthCodeOption is implemented by options for AcquireTokenByAuthCode
type AcquireByAuthCodeOption interface {
	acquireByAuthCodeOption()
}

// AcquireTokenByAuthCodeOption changes options inside AcquireTokenByAuthCodeOptions used in .AcquireTokenByAuthCode().
type AcquireTokenByAuthCodeOption func(a *AcquireTokenByAuthCodeOptions)

func (AcquireTokenByAuthCodeOption) acquireByAuthCodeOption() {}

// WithChallenge allows you to provide a challenge for the .AcquireTokenByAuthCode() call.
func WithChallenge(challenge string) interface {
	AcquireByAuthCodeOption
	options.CallOption
} {
	return struct {
		AcquireByAuthCodeOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenByAuthCodeOptions:
					t.Challenge = challenge
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// AcquireTokenByAuthCode is a request to acquire a security token from the authority, using an authorization code.
// The specified redirect URI must be the same URI that was used when the authorization code was requested.
//
// Options: [WithChallenge], [WithClaims], [WithTenantID]
func (cca Client) AcquireTokenByAuthCode(ctx context.Context, code string, redirectURI string, scopes []string, opts ...AcquireByAuthCodeOption) (AuthResult, error) {
	o := AcquireTokenByAuthCodeOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}

	params := base.AcquireTokenAuthCodeParameters{
		Scopes:      scopes,
		Code:        code,
		Challenge:   o.Challenge,
		Claims:      o.claims,
		AppType:     accesstokens.ATConfidential,
		Credential:  cca.cred, // This setting differs from public.Client.AcquireTokenByAuthCode
		RedirectURI: redirectURI,
		TenantID:    o.tenantID,
	}

	return cca.base.AcquireTokenByAuthCode(ctx, params)
}

// acquireTokenByCredentialOptions contains optional configuration for AcquireTokenByCredential
type acquireTokenByCredentialOptions struct {
	claims, tenantID string
}

// AcquireByCredentialOption is implemented by options for AcquireTokenByCredential
type AcquireByCredentialOption interface {
	acquireByCredOption()
}

// AcquireTokenByCredential acquires a security token from the authority, using the client credentials grant.
//
// Options: [WithClaims], [WithTenantID]
func (cca Client) AcquireTokenByCredential(ctx context.Context, scopes []string, opts ...AcquireByCredentialOption) (AuthResult, error) {
	o := acquireTokenByCredentialOptions{}
	err := options.ApplyOptions(&o, opts)
	if err != nil {
		return AuthResult{}, err
	}
	authParams, err := cca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return AuthResult{}, err
	}
	authParams.Scopes = scopes
	authParams.AuthorizationType = authority.ATClientCredentials
	authParams.Claims = o.claims

	token, err := cca.base.Token.Credential(ctx, authParams, cca.cred)
	if err != nil {
		return AuthResult{}, err
	}
	return cca.base.AuthResultFromToken(ctx, authParams, token, true)
}

// acquireTokenOnBehalfOfOptions contains optional configuration for AcquireTokenOnBehalfOf
type acquireTokenOnBehalfOfOptions struct {
	claims, tenantID string
}

// AcquireOnBehalfOfOption is implemented by options for AcquireTokenOnBehalfOf
type AcquireOnBehalfOfOption interface {
	acquireOBOOption()
}

// AcquireTokenOnBehalfOf acquires a security token for an app using middle tier apps access token.
// Refer https://docs.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-on-behalf-of-flow.
//
// Options: [WithClaims], [WithTenantID]
func (cca Client) AcquireTokenOnBehalfOf(ctx context.Context, userAssertion string, scopes []string, opts ...AcquireOnBehalfOfOption) (AuthResult, error) {
	o := acquireTokenOnBehalfOfOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}
	params := base.AcquireTokenOnBehalfOfParameters{
		Scopes:        scopes,
		UserAssertion: userAssertion,
		Claims:        o.claims,
		Credential:    cca.cred,
		TenantID:      o.tenantID,
	}
	return cca.base.AcquireTokenOnBehalfOf(ctx, params)
}

// Account gets the account in the token cache with the specified homeAccountID.
func (cca Client) Account(homeAccountID string) Account {
	return cca.base.Account(homeAccountID)
}

// RemoveAccount signs the account out and forgets account from token cache.
func (cca Client) RemoveAccount(account Account) error {
	cca.base.RemoveAccount(account)
	return nil
}
