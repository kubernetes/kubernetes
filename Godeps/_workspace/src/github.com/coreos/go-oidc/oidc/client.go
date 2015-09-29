package oidc

import (
	"errors"
	"fmt"
	"net/http"
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

type ClientMetadata struct {
	RedirectURLs []url.URL
}

func (m *ClientMetadata) Valid() error {
	if len(m.RedirectURLs) == 0 {
		return errors.New("zero redirect URLs")
	}

	for _, u := range m.RedirectURLs {
		if u.Scheme != "http" && u.Scheme != "https" {
			return errors.New("invalid redirect URL: scheme not http/https")
		} else if u.Host == "" {
			return errors.New("invalid redirect URL: host empty")
		}
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
		providerConfig: cfg.ProviderConfig,
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
	providerConfig ProviderConfig
	credentials    ClientCredentials
	redirectURL    string
	scope          []string
	keySet         key.PublicKeySet

	keySetSyncMutex sync.RWMutex
	lastKeySetSync  time.Time
}

func (c *Client) Healthy() error {
	now := time.Now().UTC()

	if c.providerConfig.Empty() {
		return errors.New("oidc client provider config empty")
	}

	if !c.providerConfig.ExpiresAt.IsZero() && c.providerConfig.ExpiresAt.Before(now) {
		return errors.New("oidc client provider config expired")
	}

	return nil
}

func (c *Client) OAuthClient() (*oauth2.Client, error) {
	authMethod, err := c.chooseAuthMethod()
	if err != nil {
		return nil, err
	}

	ocfg := oauth2.Config{
		Credentials: oauth2.ClientCredentials(c.credentials),
		RedirectURL: c.redirectURL,
		AuthURL:     c.providerConfig.AuthEndpoint,
		TokenURL:    c.providerConfig.TokenEndpoint,
		Scope:       c.scope,
		AuthMethod:  authMethod,
	}

	return oauth2.NewClient(c.httpClient, ocfg)
}

func (c *Client) chooseAuthMethod() (string, error) {
	if len(c.providerConfig.TokenEndpointAuthMethodsSupported) == 0 {
		return oauth2.AuthMethodClientSecretBasic, nil
	}

	for _, authMethod := range c.providerConfig.TokenEndpointAuthMethodsSupported {
		if _, ok := supportedAuthMethods[authMethod]; ok {
			return authMethod, nil
		}
	}

	return "", errors.New("no supported auth methods")
}

func (c *Client) SyncProviderConfig(discoveryURL string) chan struct{} {
	rp := &providerConfigRepo{c}
	r := NewHTTPProviderConfigGetter(c.httpClient, discoveryURL)
	return NewProviderConfigSyncer(r, rp).Run()
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

	r := NewRemotePublicKeyRepo(c.httpClient, c.providerConfig.KeysEndpoint)
	w := &clientKeyRepo{client: c}
	_, err := key.Sync(r, w)
	c.lastKeySetSync = time.Now().UTC()

	return err
}

type providerConfigRepo struct {
	client *Client
}

func (r *providerConfigRepo) Set(cfg ProviderConfig) error {
	r.client.providerConfig = cfg
	return nil
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
	if !c.providerConfig.SupportsGrantType(oauth2.GrantTypeClientCreds) {
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
		c.providerConfig.Issuer,
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
