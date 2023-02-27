// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
Package public provides a client for authentication of "public" applications. A "public"
application is defined as an app that runs on client devices (android, ios, windows, linux, ...).
These devices are "untrusted" and access resources via web APIs that must authenticate.
*/
package public

/*
Design note:

public.Client uses client.Base as an embedded type. client.Base statically assigns its attributes
during creation. As it doesn't have any pointers in it, anything borrowed from it, such as
Base.AuthParams is a copy that is free to be manipulated here.
*/

// TODO(msal): This should have example code for each method on client using Go's example doc framework.
// base usage details should be includee in the package documentation.

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/url"
	"strconv"

	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/cache"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/base"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/local"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/accesstokens"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/authority"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/options"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/shared"
	"github.com/google/uuid"
	"github.com/pkg/browser"
)

// AuthResult contains the results of one token acquisition operation.
// For details see https://aka.ms/msal-net-authenticationresult
type AuthResult = base.AuthResult

type Account = shared.Account

// Options configures the Client's behavior.
type Options struct {
	// Accessor controls cache persistence. By default there is no cache persistence.
	// This can be set with the WithCache() option.
	Accessor cache.ExportReplace

	// The host of the Azure Active Directory authority. The default is https://login.microsoftonline.com/common.
	// This can be changed with the WithAuthority() option.
	Authority string

	// The HTTP client used for making requests.
	// It defaults to a shared http.Client.
	HTTPClient ops.HTTPClient

	capabilities []string

	disableInstanceDiscovery bool
}

func (p *Options) validate() error {
	u, err := url.Parse(p.Authority)
	if err != nil {
		return fmt.Errorf("Authority options cannot be URL parsed: %w", err)
	}
	if u.Scheme != "https" {
		return fmt.Errorf("Authority(%s) did not start with https://", u.String())
	}
	return nil
}

// Option is an optional argument to the New constructor.
type Option func(o *Options)

// WithAuthority allows for a custom authority to be set. This must be a valid https url.
func WithAuthority(authority string) Option {
	return func(o *Options) {
		o.Authority = authority
	}
}

// WithCache allows you to set some type of cache for storing authentication tokens.
func WithCache(accessor cache.ExportReplace) Option {
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

// WithInstanceDiscovery set to false to disable authority validation (to support private cloud scenarios)
func WithInstanceDiscovery(enabled bool) Option {
	return func(o *Options) {
		o.disableInstanceDiscovery = !enabled
	}
}

// Client is a representation of authentication client for public applications as defined in the
// package doc. For more information, visit https://docs.microsoft.com/azure/active-directory/develop/msal-client-applications.
type Client struct {
	base base.Client
}

// New is the constructor for Client.
func New(clientID string, options ...Option) (Client, error) {
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

	base, err := base.New(clientID, opts.Authority, oauth.New(opts.HTTPClient), base.WithCacheAccessor(opts.Accessor), base.WithClientCapabilities(opts.capabilities), base.WithInstanceDiscovery(!opts.disableInstanceDiscovery))
	if err != nil {
		return Client{}, err
	}
	return Client{base}, nil
}

// createAuthCodeURLOptions contains options for CreateAuthCodeURL
type createAuthCodeURLOptions struct {
	claims, loginHint, tenantID, domainHint string
}

// CreateAuthCodeURLOption is implemented by options for CreateAuthCodeURL
type CreateAuthCodeURLOption interface {
	createAuthCodeURLOption()
}

// CreateAuthCodeURL creates a URL used to acquire an authorization code.
//
// Options: [WithClaims], [WithDomainHint], [WithLoginHint], [WithTenantID]
func (pca Client) CreateAuthCodeURL(ctx context.Context, clientID, redirectURI string, scopes []string, opts ...CreateAuthCodeURLOption) (string, error) {
	o := createAuthCodeURLOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return "", err
	}
	ap, err := pca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return "", err
	}
	ap.Claims = o.claims
	ap.LoginHint = o.loginHint
	ap.DomainHint = o.domainHint
	return pca.base.AuthCodeURL(ctx, clientID, redirectURI, scopes, ap)
}

// WithClaims sets additional claims to request for the token, such as those required by conditional access policies.
// Use this option when Azure AD returned a claims challenge for a prior request. The argument must be decoded.
// This option is valid for any token acquisition method.
func WithClaims(claims string) interface {
	AcquireByAuthCodeOption
	AcquireByDeviceCodeOption
	AcquireByUsernamePasswordOption
	AcquireInteractiveOption
	AcquireSilentOption
	CreateAuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireByAuthCodeOption
		AcquireByDeviceCodeOption
		AcquireByUsernamePasswordOption
		AcquireInteractiveOption
		AcquireSilentOption
		CreateAuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenByAuthCodeOptions:
					t.claims = claims
				case *acquireTokenByDeviceCodeOptions:
					t.claims = claims
				case *acquireTokenByUsernamePasswordOptions:
					t.claims = claims
				case *AcquireTokenSilentOptions:
					t.claims = claims
				case *createAuthCodeURLOptions:
					t.claims = claims
				case *InteractiveAuthOptions:
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
	AcquireByDeviceCodeOption
	AcquireByUsernamePasswordOption
	AcquireInteractiveOption
	AcquireSilentOption
	CreateAuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireByAuthCodeOption
		AcquireByDeviceCodeOption
		AcquireByUsernamePasswordOption
		AcquireInteractiveOption
		AcquireSilentOption
		CreateAuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *AcquireTokenByAuthCodeOptions:
					t.tenantID = tenantID
				case *acquireTokenByDeviceCodeOptions:
					t.tenantID = tenantID
				case *acquireTokenByUsernamePasswordOptions:
					t.tenantID = tenantID
				case *AcquireTokenSilentOptions:
					t.tenantID = tenantID
				case *createAuthCodeURLOptions:
					t.tenantID = tenantID
				case *InteractiveAuthOptions:
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
func (pca Client) AcquireTokenSilent(ctx context.Context, scopes []string, opts ...AcquireSilentOption) (AuthResult, error) {
	o := AcquireTokenSilentOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}

	silentParameters := base.AcquireTokenSilentParameters{
		Scopes:      scopes,
		Account:     o.Account,
		Claims:      o.claims,
		RequestType: accesstokens.ATPublic,
		IsAppCache:  false,
		TenantID:    o.tenantID,
	}

	return pca.base.AcquireTokenSilent(ctx, silentParameters)
}

// acquireTokenByUsernamePasswordOptions contains optional configuration for AcquireTokenByUsernamePassword
type acquireTokenByUsernamePasswordOptions struct {
	claims, tenantID string
}

// AcquireByUsernamePasswordOption is implemented by options for AcquireTokenByUsernamePassword
type AcquireByUsernamePasswordOption interface {
	acquireByUsernamePasswordOption()
}

// AcquireTokenByUsernamePassword acquires a security token from the authority, via Username/Password Authentication.
// NOTE: this flow is NOT recommended.
//
// Options: [WithClaims], [WithTenantID]
func (pca Client) AcquireTokenByUsernamePassword(ctx context.Context, scopes []string, username, password string, opts ...AcquireByUsernamePasswordOption) (AuthResult, error) {
	o := acquireTokenByUsernamePasswordOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}
	authParams, err := pca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return AuthResult{}, err
	}
	authParams.Scopes = scopes
	authParams.AuthorizationType = authority.ATUsernamePassword
	authParams.Claims = o.claims
	authParams.Username = username
	authParams.Password = password

	token, err := pca.base.Token.UsernamePassword(ctx, authParams)
	if err != nil {
		return AuthResult{}, err
	}
	return pca.base.AuthResultFromToken(ctx, authParams, token, true)
}

type DeviceCodeResult = accesstokens.DeviceCodeResult

// DeviceCode provides the results of the device code flows first stage (containing the code)
// that must be entered on the second device and provides a method to retrieve the AuthenticationResult
// once that code has been entered and verified.
type DeviceCode struct {
	// Result holds the information about the device code (such as the code).
	Result DeviceCodeResult

	authParams authority.AuthParams
	client     Client
	dc         oauth.DeviceCode
}

// AuthenticationResult retreives the AuthenticationResult once the user enters the code
// on the second device. Until then it blocks until the .AcquireTokenByDeviceCode() context
// is cancelled or the token expires.
func (d DeviceCode) AuthenticationResult(ctx context.Context) (AuthResult, error) {
	token, err := d.dc.Token(ctx)
	if err != nil {
		return AuthResult{}, err
	}
	return d.client.base.AuthResultFromToken(ctx, d.authParams, token, true)
}

// acquireTokenByDeviceCodeOptions contains optional configuration for AcquireTokenByDeviceCode
type acquireTokenByDeviceCodeOptions struct {
	claims, tenantID string
}

// AcquireByDeviceCodeOption is implemented by options for AcquireTokenByDeviceCode
type AcquireByDeviceCodeOption interface {
	acquireByDeviceCodeOptions()
}

// AcquireTokenByDeviceCode acquires a security token from the authority, by acquiring a device code and using that to acquire the token.
// Users need to create an AcquireTokenDeviceCodeParameters instance and pass it in.
//
// Options: [WithClaims], [WithTenantID]
func (pca Client) AcquireTokenByDeviceCode(ctx context.Context, scopes []string, opts ...AcquireByDeviceCodeOption) (DeviceCode, error) {
	o := acquireTokenByDeviceCodeOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return DeviceCode{}, err
	}
	authParams, err := pca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return DeviceCode{}, err
	}
	authParams.Scopes = scopes
	authParams.AuthorizationType = authority.ATDeviceCode
	authParams.Claims = o.claims

	dc, err := pca.base.Token.DeviceCode(ctx, authParams)
	if err != nil {
		return DeviceCode{}, err
	}

	return DeviceCode{Result: dc.Result, authParams: authParams, client: pca, dc: dc}, nil
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

// WithChallenge allows you to provide a code for the .AcquireTokenByAuthCode() call.
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
func (pca Client) AcquireTokenByAuthCode(ctx context.Context, code string, redirectURI string, scopes []string, opts ...AcquireByAuthCodeOption) (AuthResult, error) {
	o := AcquireTokenByAuthCodeOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}

	params := base.AcquireTokenAuthCodeParameters{
		Scopes:      scopes,
		Code:        code,
		Challenge:   o.Challenge,
		Claims:      o.claims,
		AppType:     accesstokens.ATPublic,
		RedirectURI: redirectURI,
		TenantID:    o.tenantID,
	}

	return pca.base.AcquireTokenByAuthCode(ctx, params)
}

// Accounts gets all the accounts in the token cache.
// If there are no accounts in the cache the returned slice is empty.
func (pca Client) Accounts() []Account {
	return pca.base.AllAccounts()
}

// RemoveAccount signs the account out and forgets account from token cache.
func (pca Client) RemoveAccount(account Account) error {
	pca.base.RemoveAccount(account)
	return nil
}

// InteractiveAuthOptions contains the optional parameters used to acquire an access token for interactive auth code flow.
type InteractiveAuthOptions struct {
	// Used to specify a custom port for the local server.  http://localhost:portnumber
	// All other URI components are ignored.
	RedirectURI string

	claims, loginHint, tenantID, domainHint string
}

// AcquireInteractiveOption is implemented by options for AcquireTokenInteractive
type AcquireInteractiveOption interface {
	acquireInteractiveOption()
}

// InteractiveAuthOption changes options inside InteractiveAuthOptions used in .AcquireTokenInteractive().
type InteractiveAuthOption func(*InteractiveAuthOptions)

func (InteractiveAuthOption) acquireInteractiveOption() {}

// WithLoginHint pre-populates the login prompt with a username.
func WithLoginHint(username string) interface {
	AcquireInteractiveOption
	CreateAuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireInteractiveOption
		CreateAuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *createAuthCodeURLOptions:
					t.loginHint = username
				case *InteractiveAuthOptions:
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
	AcquireInteractiveOption
	CreateAuthCodeURLOption
	options.CallOption
} {
	return struct {
		AcquireInteractiveOption
		CreateAuthCodeURLOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *createAuthCodeURLOptions:
					t.domainHint = domain
				case *InteractiveAuthOptions:
					t.domainHint = domain
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// WithRedirectURI uses the specified redirect URI for interactive auth.
func WithRedirectURI(redirectURI string) interface {
	AcquireInteractiveOption
	options.CallOption
} {
	return struct {
		AcquireInteractiveOption
		options.CallOption
	}{
		CallOption: options.NewCallOption(
			func(a any) error {
				switch t := a.(type) {
				case *InteractiveAuthOptions:
					t.RedirectURI = redirectURI
				default:
					return fmt.Errorf("unexpected options type %T", a)
				}
				return nil
			},
		),
	}
}

// AcquireTokenInteractive acquires a security token from the authority using the default web browser to select the account.
// https://docs.microsoft.com/en-us/azure/active-directory/develop/msal-authentication-flows#interactive-and-non-interactive-authentication
//
// Options: [WithDomainHint], [WithLoginHint], [WithRedirectURI], [WithTenantID]
func (pca Client) AcquireTokenInteractive(ctx context.Context, scopes []string, opts ...AcquireInteractiveOption) (AuthResult, error) {
	o := InteractiveAuthOptions{}
	if err := options.ApplyOptions(&o, opts); err != nil {
		return AuthResult{}, err
	}
	// the code verifier is a random 32-byte sequence that's been base-64 encoded without padding.
	// it's used to prevent MitM attacks during auth code flow, see https://tools.ietf.org/html/rfc7636
	cv, challenge, err := codeVerifier()
	if err != nil {
		return AuthResult{}, err
	}
	var redirectURL *url.URL
	if o.RedirectURI != "" {
		redirectURL, err = url.Parse(o.RedirectURI)
		if err != nil {
			return AuthResult{}, err
		}
	}
	authParams, err := pca.base.AuthParams.WithTenant(o.tenantID)
	if err != nil {
		return AuthResult{}, err
	}
	authParams.Scopes = scopes
	authParams.AuthorizationType = authority.ATInteractive
	authParams.Claims = o.claims
	authParams.CodeChallenge = challenge
	authParams.CodeChallengeMethod = "S256"
	authParams.LoginHint = o.loginHint
	authParams.DomainHint = o.domainHint
	authParams.State = uuid.New().String()
	authParams.Prompt = "select_account"
	res, err := pca.browserLogin(ctx, redirectURL, authParams)
	if err != nil {
		return AuthResult{}, err
	}
	authParams.Redirecturi = res.redirectURI

	req, err := accesstokens.NewCodeChallengeRequest(authParams, accesstokens.ATPublic, nil, res.authCode, cv)
	if err != nil {
		return AuthResult{}, err
	}

	token, err := pca.base.Token.AuthCode(ctx, req)
	if err != nil {
		return AuthResult{}, err
	}

	return pca.base.AuthResultFromToken(ctx, authParams, token, true)
}

type interactiveAuthResult struct {
	authCode    string
	redirectURI string
}

// provides a test hook to simulate opening a browser
var browserOpenURL = func(authURL string) error {
	return browser.OpenURL(authURL)
}

// parses the port number from the provided URL.
// returns 0 if nil or no port is specified.
func parsePort(u *url.URL) (int, error) {
	if u == nil {
		return 0, nil
	}
	p := u.Port()
	if p == "" {
		return 0, nil
	}
	return strconv.Atoi(p)
}

// browserLogin launches the system browser for interactive login
func (pca Client) browserLogin(ctx context.Context, redirectURI *url.URL, params authority.AuthParams) (interactiveAuthResult, error) {
	// start local redirect server so login can call us back
	port, err := parsePort(redirectURI)
	if err != nil {
		return interactiveAuthResult{}, err
	}
	srv, err := local.New(params.State, port)
	if err != nil {
		return interactiveAuthResult{}, err
	}
	defer srv.Shutdown()
	params.Scopes = accesstokens.AppendDefaultScopes(params)
	authURL, err := pca.base.AuthCodeURL(ctx, params.ClientID, srv.Addr, params.Scopes, params)
	if err != nil {
		return interactiveAuthResult{}, err
	}
	// open browser window so user can select credentials
	if err := browserOpenURL(authURL); err != nil {
		return interactiveAuthResult{}, err
	}
	// now wait until the logic calls us back
	res := srv.Result(ctx)
	if res.Err != nil {
		return interactiveAuthResult{}, res.Err
	}
	return interactiveAuthResult{
		authCode:    res.Code,
		redirectURI: srv.Addr,
	}, nil
}

// creates a code verifier string along with its SHA256 hash which
// is used as the challenge when requesting an auth code.
// used in interactive auth flow for PKCE.
func codeVerifier() (codeVerifier string, challenge string, err error) {
	cvBytes := make([]byte, 32)
	if _, err = rand.Read(cvBytes); err != nil {
		return
	}
	codeVerifier = base64.RawURLEncoding.EncodeToString(cvBytes)
	// for PKCE, create a hash of the code verifier
	cvh := sha256.Sum256([]byte(codeVerifier))
	challenge = base64.RawURLEncoding.EncodeToString(cvh[:])
	return
}
