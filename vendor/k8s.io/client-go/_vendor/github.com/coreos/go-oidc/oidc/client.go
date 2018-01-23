package oidc

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/mail"
	"net/url"
	"sync"
	"time"

	phttp "github.com/coreos/go-oidc/http"
	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oauth2"
)

const (
	// amount of time that must pass after the last key sync
	// completes before another attempt may begin
	keySyncWindow = 5 * time.Second
)

var (
	DefaultScope = []string{"openid", "email", "profile"}

	supportedAuthMethods = map[string]struct{}{
		oauth2.AuthMethodClientSecretBasic: struct{}{},
		oauth2.AuthMethodClientSecretPost:  struct{}{},
	}
)

type ClientCredentials oauth2.ClientCredentials

type ClientIdentity struct {
	Credentials ClientCredentials
	Metadata    ClientMetadata
}

type JWAOptions struct {
	// SigningAlg specifies an JWA alg for signing JWTs.
	//
	// Specifying this field implies different actions depending on the context. It may
	// require objects be serialized and signed as a JWT instead of plain JSON, or
	// require an existing JWT object use the specified alg.
	//
	// See: http://openid.net/specs/openid-connect-registration-1_0.html#ClientMetadata
	SigningAlg string
	// EncryptionAlg, if provided, specifies that the returned or sent object be stored
	// (or nested) within a JWT object and encrypted with the provided JWA alg.
	EncryptionAlg string
	// EncryptionEnc specifies the JWA enc algorithm to use with EncryptionAlg. If
	// EncryptionAlg is provided and EncryptionEnc is omitted, this field defaults
	// to A128CBC-HS256.
	//
	// If EncryptionEnc is provided EncryptionAlg must also be specified.
	EncryptionEnc string
}

func (opt JWAOptions) valid() error {
	if opt.EncryptionEnc != "" && opt.EncryptionAlg == "" {
		return errors.New("encryption encoding provided with no encryption algorithm")
	}
	return nil
}

func (opt JWAOptions) defaults() JWAOptions {
	if opt.EncryptionAlg != "" && opt.EncryptionEnc == "" {
		opt.EncryptionEnc = jose.EncA128CBCHS256
	}
	return opt
}

var (
	// Ensure ClientMetadata satisfies these interfaces.
	_ json.Marshaler   = &ClientMetadata{}
	_ json.Unmarshaler = &ClientMetadata{}
)

// ClientMetadata holds metadata that the authorization server associates
// with a client identifier. The fields range from human-facing display
// strings such as client name, to items that impact the security of the
// protocol, such as the list of valid redirect URIs.
//
// See http://openid.net/specs/openid-connect-registration-1_0.html#ClientMetadata
//
// TODO: support language specific claim representations
// http://openid.net/specs/openid-connect-registration-1_0.html#LanguagesAndScripts
type ClientMetadata struct {
	RedirectURIs []url.URL // Required

	// A list of OAuth 2.0 "response_type" values that the client wishes to restrict
	// itself to. Either "code", "token", or another registered extension.
	//
	// If omitted, only "code" will be used.
	ResponseTypes []string
	// A list of OAuth 2.0 grant types the client wishes to restrict itself to.
	// The grant type values used by OIDC are "authorization_code", "implicit",
	// and "refresh_token".
	//
	// If ommitted, only "authorization_code" will be used.
	GrantTypes []string
	// "native" or "web". If omitted, "web".
	ApplicationType string

	// List of email addresses.
	Contacts []mail.Address
	// Name of client to be presented to the end-user.
	ClientName string
	// URL that references a logo for the Client application.
	LogoURI *url.URL
	// URL of the home page of the Client.
	ClientURI *url.URL
	// Profile data policies and terms of use to be provided to the end user.
	PolicyURI         *url.URL
	TermsOfServiceURI *url.URL

	// URL to or the value of the client's JSON Web Key Set document.
	JWKSURI *url.URL
	JWKS    *jose.JWKSet

	// URL referencing a flie with a single JSON array of redirect URIs.
	SectorIdentifierURI *url.URL

	SubjectType string

	// Options to restrict the JWS alg and enc values used for server responses and requests.
	IDTokenResponseOptions  JWAOptions
	UserInfoResponseOptions JWAOptions
	RequestObjectOptions    JWAOptions

	// Client requested authorization method and signing options for the token endpoint.
	//
	// Defaults to "client_secret_basic"
	TokenEndpointAuthMethod     string
	TokenEndpointAuthSigningAlg string

	// DefaultMaxAge specifies the maximum amount of time in seconds before an authorized
	// user must reauthroize.
	//
	// If 0, no limitation is placed on the maximum.
	DefaultMaxAge int64
	// RequireAuthTime specifies if the auth_time claim in the ID token is required.
	RequireAuthTime bool

	// Default Authentication Context Class Reference values for authentication requests.
	DefaultACRValues []string

	// URI that a third party can use to initiate a login by the relaying party.
	//
	// See: http://openid.net/specs/openid-connect-core-1_0.html#ThirdPartyInitiatedLogin
	InitiateLoginURI *url.URL
	// Pre-registered request_uri values that may be cached by the server.
	RequestURIs []url.URL
}

// Defaults returns a shallow copy of ClientMetadata with default
// values replacing omitted fields.
func (m ClientMetadata) Defaults() ClientMetadata {
	if len(m.ResponseTypes) == 0 {
		m.ResponseTypes = []string{oauth2.ResponseTypeCode}
	}
	if len(m.GrantTypes) == 0 {
		m.GrantTypes = []string{oauth2.GrantTypeAuthCode}
	}
	if m.ApplicationType == "" {
		m.ApplicationType = "web"
	}
	if m.TokenEndpointAuthMethod == "" {
		m.TokenEndpointAuthMethod = oauth2.AuthMethodClientSecretBasic
	}
	m.IDTokenResponseOptions = m.IDTokenResponseOptions.defaults()
	m.UserInfoResponseOptions = m.UserInfoResponseOptions.defaults()
	m.RequestObjectOptions = m.RequestObjectOptions.defaults()
	return m
}

func (m *ClientMetadata) MarshalJSON() ([]byte, error) {
	e := m.toEncodableStruct()
	return json.Marshal(&e)
}

func (m *ClientMetadata) UnmarshalJSON(data []byte) error {
	var e encodableClientMetadata
	if err := json.Unmarshal(data, &e); err != nil {
		return err
	}
	meta, err := e.toStruct()
	if err != nil {
		return err
	}
	if err := meta.Valid(); err != nil {
		return err
	}
	*m = meta
	return nil
}

type encodableClientMetadata struct {
	RedirectURIs                 []string     `json:"redirect_uris"` // Required
	ResponseTypes                []string     `json:"response_types,omitempty"`
	GrantTypes                   []string     `json:"grant_types,omitempty"`
	ApplicationType              string       `json:"application_type,omitempty"`
	Contacts                     []string     `json:"contacts,omitempty"`
	ClientName                   string       `json:"client_name,omitempty"`
	LogoURI                      string       `json:"logo_uri,omitempty"`
	ClientURI                    string       `json:"client_uri,omitempty"`
	PolicyURI                    string       `json:"policy_uri,omitempty"`
	TermsOfServiceURI            string       `json:"tos_uri,omitempty"`
	JWKSURI                      string       `json:"jwks_uri,omitempty"`
	JWKS                         *jose.JWKSet `json:"jwks,omitempty"`
	SectorIdentifierURI          string       `json:"sector_identifier_uri,omitempty"`
	SubjectType                  string       `json:"subject_type,omitempty"`
	IDTokenSignedResponseAlg     string       `json:"id_token_signed_response_alg,omitempty"`
	IDTokenEncryptedResponseAlg  string       `json:"id_token_encrypted_response_alg,omitempty"`
	IDTokenEncryptedResponseEnc  string       `json:"id_token_encrypted_response_enc,omitempty"`
	UserInfoSignedResponseAlg    string       `json:"userinfo_signed_response_alg,omitempty"`
	UserInfoEncryptedResponseAlg string       `json:"userinfo_encrypted_response_alg,omitempty"`
	UserInfoEncryptedResponseEnc string       `json:"userinfo_encrypted_response_enc,omitempty"`
	RequestObjectSigningAlg      string       `json:"request_object_signing_alg,omitempty"`
	RequestObjectEncryptionAlg   string       `json:"request_object_encryption_alg,omitempty"`
	RequestObjectEncryptionEnc   string       `json:"request_object_encryption_enc,omitempty"`
	TokenEndpointAuthMethod      string       `json:"token_endpoint_auth_method,omitempty"`
	TokenEndpointAuthSigningAlg  string       `json:"token_endpoint_auth_signing_alg,omitempty"`
	DefaultMaxAge                int64        `json:"default_max_age,omitempty"`
	RequireAuthTime              bool         `json:"require_auth_time,omitempty"`
	DefaultACRValues             []string     `json:"default_acr_values,omitempty"`
	InitiateLoginURI             string       `json:"initiate_login_uri,omitempty"`
	RequestURIs                  []string     `json:"request_uris,omitempty"`
}

func (c *encodableClientMetadata) toStruct() (ClientMetadata, error) {
	p := stickyErrParser{}
	m := ClientMetadata{
		RedirectURIs:                p.parseURIs(c.RedirectURIs, "redirect_uris"),
		ResponseTypes:               c.ResponseTypes,
		GrantTypes:                  c.GrantTypes,
		ApplicationType:             c.ApplicationType,
		Contacts:                    p.parseEmails(c.Contacts, "contacts"),
		ClientName:                  c.ClientName,
		LogoURI:                     p.parseURI(c.LogoURI, "logo_uri"),
		ClientURI:                   p.parseURI(c.ClientURI, "client_uri"),
		PolicyURI:                   p.parseURI(c.PolicyURI, "policy_uri"),
		TermsOfServiceURI:           p.parseURI(c.TermsOfServiceURI, "tos_uri"),
		JWKSURI:                     p.parseURI(c.JWKSURI, "jwks_uri"),
		JWKS:                        c.JWKS,
		SectorIdentifierURI:         p.parseURI(c.SectorIdentifierURI, "sector_identifier_uri"),
		SubjectType:                 c.SubjectType,
		TokenEndpointAuthMethod:     c.TokenEndpointAuthMethod,
		TokenEndpointAuthSigningAlg: c.TokenEndpointAuthSigningAlg,
		DefaultMaxAge:               c.DefaultMaxAge,
		RequireAuthTime:             c.RequireAuthTime,
		DefaultACRValues:            c.DefaultACRValues,
		InitiateLoginURI:            p.parseURI(c.InitiateLoginURI, "initiate_login_uri"),
		RequestURIs:                 p.parseURIs(c.RequestURIs, "request_uris"),
		IDTokenResponseOptions: JWAOptions{
			c.IDTokenSignedResponseAlg,
			c.IDTokenEncryptedResponseAlg,
			c.IDTokenEncryptedResponseEnc,
		},
		UserInfoResponseOptions: JWAOptions{
			c.UserInfoSignedResponseAlg,
			c.UserInfoEncryptedResponseAlg,
			c.UserInfoEncryptedResponseEnc,
		},
		RequestObjectOptions: JWAOptions{
			c.RequestObjectSigningAlg,
			c.RequestObjectEncryptionAlg,
			c.RequestObjectEncryptionEnc,
		},
	}
	if p.firstErr != nil {
		return ClientMetadata{}, p.firstErr
	}
	return m, nil
}

// stickyErrParser parses URIs and email addresses. Once it encounters
// a parse error, subsequent calls become no-op.
type stickyErrParser struct {
	firstErr error
}

func (p *stickyErrParser) parseURI(s, field string) *url.URL {
	if p.firstErr != nil || s == "" {
		return nil
	}
	u, err := url.Parse(s)
	if err == nil {
		if u.Host == "" {
			err = errors.New("no host in URI")
		} else if u.Scheme != "http" && u.Scheme != "https" {
			err = errors.New("invalid URI scheme")
		}
	}
	if err != nil {
		p.firstErr = fmt.Errorf("failed to parse %s: %v", field, err)
		return nil
	}
	return u
}

func (p *stickyErrParser) parseURIs(s []string, field string) []url.URL {
	if p.firstErr != nil || len(s) == 0 {
		return nil
	}
	uris := make([]url.URL, len(s))
	for i, val := range s {
		if val == "" {
			p.firstErr = fmt.Errorf("invalid URI in field %s", field)
			return nil
		}
		if u := p.parseURI(val, field); u != nil {
			uris[i] = *u
		}
	}
	return uris
}

func (p *stickyErrParser) parseEmails(s []string, field string) []mail.Address {
	if p.firstErr != nil || len(s) == 0 {
		return nil
	}
	addrs := make([]mail.Address, len(s))
	for i, addr := range s {
		if addr == "" {
			p.firstErr = fmt.Errorf("invalid email in field %s", field)
			return nil
		}
		a, err := mail.ParseAddress(addr)
		if err != nil {
			p.firstErr = fmt.Errorf("invalid email in field %s: %v", field, err)
			return nil
		}
		addrs[i] = *a
	}
	return addrs
}

func (m *ClientMetadata) toEncodableStruct() encodableClientMetadata {
	return encodableClientMetadata{
		RedirectURIs:                 urisToStrings(m.RedirectURIs),
		ResponseTypes:                m.ResponseTypes,
		GrantTypes:                   m.GrantTypes,
		ApplicationType:              m.ApplicationType,
		Contacts:                     emailsToStrings(m.Contacts),
		ClientName:                   m.ClientName,
		LogoURI:                      uriToString(m.LogoURI),
		ClientURI:                    uriToString(m.ClientURI),
		PolicyURI:                    uriToString(m.PolicyURI),
		TermsOfServiceURI:            uriToString(m.TermsOfServiceURI),
		JWKSURI:                      uriToString(m.JWKSURI),
		JWKS:                         m.JWKS,
		SectorIdentifierURI:          uriToString(m.SectorIdentifierURI),
		SubjectType:                  m.SubjectType,
		IDTokenSignedResponseAlg:     m.IDTokenResponseOptions.SigningAlg,
		IDTokenEncryptedResponseAlg:  m.IDTokenResponseOptions.EncryptionAlg,
		IDTokenEncryptedResponseEnc:  m.IDTokenResponseOptions.EncryptionEnc,
		UserInfoSignedResponseAlg:    m.UserInfoResponseOptions.SigningAlg,
		UserInfoEncryptedResponseAlg: m.UserInfoResponseOptions.EncryptionAlg,
		UserInfoEncryptedResponseEnc: m.UserInfoResponseOptions.EncryptionEnc,
		RequestObjectSigningAlg:      m.RequestObjectOptions.SigningAlg,
		RequestObjectEncryptionAlg:   m.RequestObjectOptions.EncryptionAlg,
		RequestObjectEncryptionEnc:   m.RequestObjectOptions.EncryptionEnc,
		TokenEndpointAuthMethod:      m.TokenEndpointAuthMethod,
		TokenEndpointAuthSigningAlg:  m.TokenEndpointAuthSigningAlg,
		DefaultMaxAge:                m.DefaultMaxAge,
		RequireAuthTime:              m.RequireAuthTime,
		DefaultACRValues:             m.DefaultACRValues,
		InitiateLoginURI:             uriToString(m.InitiateLoginURI),
		RequestURIs:                  urisToStrings(m.RequestURIs),
	}
}

func uriToString(u *url.URL) string {
	if u == nil {
		return ""
	}
	return u.String()
}

func urisToStrings(urls []url.URL) []string {
	if len(urls) == 0 {
		return nil
	}
	sli := make([]string, len(urls))
	for i, u := range urls {
		sli[i] = u.String()
	}
	return sli
}

func emailsToStrings(addrs []mail.Address) []string {
	if len(addrs) == 0 {
		return nil
	}
	sli := make([]string, len(addrs))
	for i, addr := range addrs {
		sli[i] = addr.String()
	}
	return sli
}

// Valid determines if a ClientMetadata conforms with the OIDC specification.
//
// Valid is called by UnmarshalJSON.
//
// NOTE(ericchiang): For development purposes Valid does not mandate 'https' for
// URLs fields where the OIDC spec requires it. This may change in future releases
// of this package. See: https://github.com/coreos/go-oidc/issues/34
func (m *ClientMetadata) Valid() error {
	if len(m.RedirectURIs) == 0 {
		return errors.New("zero redirect URLs")
	}

	validURI := func(u *url.URL, fieldName string) error {
		if u.Host == "" {
			return fmt.Errorf("no host for uri field %s", fieldName)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return fmt.Errorf("uri field %s scheme is not http or https", fieldName)
		}
		return nil
	}

	uris := []struct {
		val  *url.URL
		name string
	}{
		{m.LogoURI, "logo_uri"},
		{m.ClientURI, "client_uri"},
		{m.PolicyURI, "policy_uri"},
		{m.TermsOfServiceURI, "tos_uri"},
		{m.JWKSURI, "jwks_uri"},
		{m.SectorIdentifierURI, "sector_identifier_uri"},
		{m.InitiateLoginURI, "initiate_login_uri"},
	}

	for _, uri := range uris {
		if uri.val == nil {
			continue
		}
		if err := validURI(uri.val, uri.name); err != nil {
			return err
		}
	}

	uriLists := []struct {
		vals []url.URL
		name string
	}{
		{m.RedirectURIs, "redirect_uris"},
		{m.RequestURIs, "request_uris"},
	}
	for _, list := range uriLists {
		for _, uri := range list.vals {
			if err := validURI(&uri, list.name); err != nil {
				return err
			}
		}
	}

	options := []struct {
		option JWAOptions
		name   string
	}{
		{m.IDTokenResponseOptions, "id_token response"},
		{m.UserInfoResponseOptions, "userinfo response"},
		{m.RequestObjectOptions, "request_object"},
	}
	for _, option := range options {
		if err := option.option.valid(); err != nil {
			return fmt.Errorf("invalid JWA values for %s: %v", option.name, err)
		}
	}
	return nil
}

type ClientRegistrationResponse struct {
	ClientID                string // Required
	ClientSecret            string
	RegistrationAccessToken string
	RegistrationClientURI   string
	// If IsZero is true, unspecified.
	ClientIDIssuedAt time.Time
	// Time at which the client_secret will expire.
	// If IsZero is true, it will not expire.
	ClientSecretExpiresAt time.Time

	ClientMetadata
}

type encodableClientRegistrationResponse struct {
	ClientID                string `json:"client_id"` // Required
	ClientSecret            string `json:"client_secret,omitempty"`
	RegistrationAccessToken string `json:"registration_access_token,omitempty"`
	RegistrationClientURI   string `json:"registration_client_uri,omitempty"`
	ClientIDIssuedAt        int64  `json:"client_id_issued_at,omitempty"`
	// Time at which the client_secret will expire, in seconds since the epoch.
	// If 0 it will not expire.
	ClientSecretExpiresAt int64 `json:"client_secret_expires_at"` // Required

	encodableClientMetadata
}

func unixToSec(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.Unix()
}

func (c *ClientRegistrationResponse) MarshalJSON() ([]byte, error) {
	e := encodableClientRegistrationResponse{
		ClientID:                c.ClientID,
		ClientSecret:            c.ClientSecret,
		RegistrationAccessToken: c.RegistrationAccessToken,
		RegistrationClientURI:   c.RegistrationClientURI,
		ClientIDIssuedAt:        unixToSec(c.ClientIDIssuedAt),
		ClientSecretExpiresAt:   unixToSec(c.ClientSecretExpiresAt),
		encodableClientMetadata: c.ClientMetadata.toEncodableStruct(),
	}
	return json.Marshal(&e)
}

func secToUnix(sec int64) time.Time {
	if sec == 0 {
		return time.Time{}
	}
	return time.Unix(sec, 0)
}

func (c *ClientRegistrationResponse) UnmarshalJSON(data []byte) error {
	var e encodableClientRegistrationResponse
	if err := json.Unmarshal(data, &e); err != nil {
		return err
	}
	if e.ClientID == "" {
		return errors.New("no client_id in client registration response")
	}
	metadata, err := e.encodableClientMetadata.toStruct()
	if err != nil {
		return err
	}
	*c = ClientRegistrationResponse{
		ClientID:                e.ClientID,
		ClientSecret:            e.ClientSecret,
		RegistrationAccessToken: e.RegistrationAccessToken,
		RegistrationClientURI:   e.RegistrationClientURI,
		ClientIDIssuedAt:        secToUnix(e.ClientIDIssuedAt),
		ClientSecretExpiresAt:   secToUnix(e.ClientSecretExpiresAt),
		ClientMetadata:          metadata,
	}
	return nil
}

type ClientConfig struct {
	HTTPClient     phttp.Client
	Credentials    ClientCredentials
	Scope          []string
	RedirectURL    string
	ProviderConfig ProviderConfig
	KeySet         key.PublicKeySet
}

func NewClient(cfg ClientConfig) (*Client, error) {
	// Allow empty redirect URL in the case where the client
	// only needs to verify a given token.
	ru, err := url.Parse(cfg.RedirectURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redirect URL: %v", err)
	}

	c := Client{
		credentials:    cfg.Credentials,
		httpClient:     cfg.HTTPClient,
		scope:          cfg.Scope,
		redirectURL:    ru.String(),
		providerConfig: newProviderConfigRepo(cfg.ProviderConfig),
		keySet:         cfg.KeySet,
	}

	if c.httpClient == nil {
		c.httpClient = http.DefaultClient
	}

	if c.scope == nil {
		c.scope = make([]string, len(DefaultScope))
		copy(c.scope, DefaultScope)
	}

	return &c, nil
}

type Client struct {
	httpClient     phttp.Client
	providerConfig *providerConfigRepo
	credentials    ClientCredentials
	redirectURL    string
	scope          []string
	keySet         key.PublicKeySet
	providerSyncer *ProviderConfigSyncer

	keySetSyncMutex sync.RWMutex
	lastKeySetSync  time.Time
}

func (c *Client) Healthy() error {
	now := time.Now().UTC()

	cfg := c.providerConfig.Get()

	if cfg.Empty() {
		return errors.New("oidc client provider config empty")
	}

	if !cfg.ExpiresAt.IsZero() && cfg.ExpiresAt.Before(now) {
		return errors.New("oidc client provider config expired")
	}

	return nil
}

func (c *Client) OAuthClient() (*oauth2.Client, error) {
	cfg := c.providerConfig.Get()
	authMethod, err := chooseAuthMethod(cfg)
	if err != nil {
		return nil, err
	}

	ocfg := oauth2.Config{
		Credentials: oauth2.ClientCredentials(c.credentials),
		RedirectURL: c.redirectURL,
		AuthURL:     cfg.AuthEndpoint.String(),
		TokenURL:    cfg.TokenEndpoint.String(),
		Scope:       c.scope,
		AuthMethod:  authMethod,
	}

	return oauth2.NewClient(c.httpClient, ocfg)
}

func chooseAuthMethod(cfg ProviderConfig) (string, error) {
	if len(cfg.TokenEndpointAuthMethodsSupported) == 0 {
		return oauth2.AuthMethodClientSecretBasic, nil
	}

	for _, authMethod := range cfg.TokenEndpointAuthMethodsSupported {
		if _, ok := supportedAuthMethods[authMethod]; ok {
			return authMethod, nil
		}
	}

	return "", errors.New("no supported auth methods")
}

// SyncProviderConfig starts the provider config syncer
func (c *Client) SyncProviderConfig(discoveryURL string) chan struct{} {
	r := NewHTTPProviderConfigGetter(c.httpClient, discoveryURL)
	s := NewProviderConfigSyncer(r, c.providerConfig)
	stop := s.Run()
	s.WaitUntilInitialSync()
	return stop
}

func (c *Client) maybeSyncKeys() error {
	tooSoon := func() bool {
		return time.Now().UTC().Before(c.lastKeySetSync.Add(keySyncWindow))
	}

	// ignore request to sync keys if a sync operation has been
	// attempted too recently
	if tooSoon() {
		return nil
	}

	c.keySetSyncMutex.Lock()
	defer c.keySetSyncMutex.Unlock()

	// check again, as another goroutine may have been holding
	// the lock while updating the keys
	if tooSoon() {
		return nil
	}

	cfg := c.providerConfig.Get()
	r := NewRemotePublicKeyRepo(c.httpClient, cfg.KeysEndpoint.String())
	w := &clientKeyRepo{client: c}
	_, err := key.Sync(r, w)
	c.lastKeySetSync = time.Now().UTC()

	return err
}

type clientKeyRepo struct {
	client *Client
}

func (r *clientKeyRepo) Set(ks key.KeySet) error {
	pks, ok := ks.(*key.PublicKeySet)
	if !ok {
		return errors.New("unable to cast to PublicKey")
	}
	r.client.keySet = *pks
	return nil
}

func (c *Client) ClientCredsToken(scope []string) (jose.JWT, error) {
	cfg := c.providerConfig.Get()

	if !cfg.SupportsGrantType(oauth2.GrantTypeClientCreds) {
		return jose.JWT{}, fmt.Errorf("%v grant type is not supported", oauth2.GrantTypeClientCreds)
	}

	oac, err := c.OAuthClient()
	if err != nil {
		return jose.JWT{}, err
	}

	t, err := oac.ClientCredsToken(scope)
	if err != nil {
		return jose.JWT{}, err
	}

	jwt, err := jose.ParseJWT(t.IDToken)
	if err != nil {
		return jose.JWT{}, err
	}

	return jwt, c.VerifyJWT(jwt)
}

// ExchangeAuthCode exchanges an OAuth2 auth code for an OIDC JWT ID token.
func (c *Client) ExchangeAuthCode(code string) (jose.JWT, error) {
	oac, err := c.OAuthClient()
	if err != nil {
		return jose.JWT{}, err
	}

	t, err := oac.RequestToken(oauth2.GrantTypeAuthCode, code)
	if err != nil {
		return jose.JWT{}, err
	}

	jwt, err := jose.ParseJWT(t.IDToken)
	if err != nil {
		return jose.JWT{}, err
	}

	return jwt, c.VerifyJWT(jwt)
}

// RefreshToken uses a refresh token to exchange for a new OIDC JWT ID Token.
func (c *Client) RefreshToken(refreshToken string) (jose.JWT, error) {
	oac, err := c.OAuthClient()
	if err != nil {
		return jose.JWT{}, err
	}

	t, err := oac.RequestToken(oauth2.GrantTypeRefreshToken, refreshToken)
	if err != nil {
		return jose.JWT{}, err
	}

	jwt, err := jose.ParseJWT(t.IDToken)
	if err != nil {
		return jose.JWT{}, err
	}

	return jwt, c.VerifyJWT(jwt)
}

func (c *Client) VerifyJWT(jwt jose.JWT) error {
	var keysFunc func() []key.PublicKey
	if kID, ok := jwt.KeyID(); ok {
		keysFunc = c.keysFuncWithID(kID)
	} else {
		keysFunc = c.keysFuncAll()
	}

	v := NewJWTVerifier(
		c.providerConfig.Get().Issuer.String(),
		c.credentials.ID,
		c.maybeSyncKeys, keysFunc)

	return v.Verify(jwt)
}

// keysFuncWithID returns a function that retrieves at most unexpired
// public key from the Client that matches the provided ID
func (c *Client) keysFuncWithID(kID string) func() []key.PublicKey {
	return func() []key.PublicKey {
		c.keySetSyncMutex.RLock()
		defer c.keySetSyncMutex.RUnlock()

		if c.keySet.ExpiresAt().Before(time.Now()) {
			return []key.PublicKey{}
		}

		k := c.keySet.Key(kID)
		if k == nil {
			return []key.PublicKey{}
		}

		return []key.PublicKey{*k}
	}
}

// keysFuncAll returns a function that retrieves all unexpired public
// keys from the Client
func (c *Client) keysFuncAll() func() []key.PublicKey {
	return func() []key.PublicKey {
		c.keySetSyncMutex.RLock()
		defer c.keySetSyncMutex.RUnlock()

		if c.keySet.ExpiresAt().Before(time.Now()) {
			return []key.PublicKey{}
		}

		return c.keySet.Keys()
	}
}

type providerConfigRepo struct {
	mu     sync.RWMutex
	config ProviderConfig // do not access directly, use Get()
}

func newProviderConfigRepo(pc ProviderConfig) *providerConfigRepo {
	return &providerConfigRepo{sync.RWMutex{}, pc}
}

// returns an error to implement ProviderConfigSetter
func (r *providerConfigRepo) Set(cfg ProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config = cfg
	return nil
}

func (r *providerConfigRepo) Get() ProviderConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.config
}
