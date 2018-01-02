package auth

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/docker/distribution/registry/client"
	"github.com/docker/distribution/registry/client/auth/challenge"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/sirupsen/logrus"
)

var (
	// ErrNoBasicAuthCredentials is returned if a request can't be authorized with
	// basic auth due to lack of credentials.
	ErrNoBasicAuthCredentials = errors.New("no basic auth credentials")

	// ErrNoToken is returned if a request is successful but the body does not
	// contain an authorization token.
	ErrNoToken = errors.New("authorization server did not include a token in the response")
)

const defaultClientID = "registry-client"

// AuthenticationHandler is an interface for authorizing a request from
// params from a "WWW-Authenicate" header for a single scheme.
type AuthenticationHandler interface {
	// Scheme returns the scheme as expected from the "WWW-Authenicate" header.
	Scheme() string

	// AuthorizeRequest adds the authorization header to a request (if needed)
	// using the parameters from "WWW-Authenticate" method. The parameters
	// values depend on the scheme.
	AuthorizeRequest(req *http.Request, params map[string]string) error
}

// CredentialStore is an interface for getting credentials for
// a given URL
type CredentialStore interface {
	// Basic returns basic auth for the given URL
	Basic(*url.URL) (string, string)

	// RefreshToken returns a refresh token for the
	// given URL and service
	RefreshToken(*url.URL, string) string

	// SetRefreshToken sets the refresh token if none
	// is provided for the given url and service
	SetRefreshToken(realm *url.URL, service, token string)
}

// NewAuthorizer creates an authorizer which can handle multiple authentication
// schemes. The handlers are tried in order, the higher priority authentication
// methods should be first. The challengeMap holds a list of challenges for
// a given root API endpoint (for example "https://registry-1.docker.io/v2/").
func NewAuthorizer(manager challenge.Manager, handlers ...AuthenticationHandler) transport.RequestModifier {
	return &endpointAuthorizer{
		challenges: manager,
		handlers:   handlers,
	}
}

type endpointAuthorizer struct {
	challenges challenge.Manager
	handlers   []AuthenticationHandler
	transport  http.RoundTripper
}

func (ea *endpointAuthorizer) ModifyRequest(req *http.Request) error {
	pingPath := req.URL.Path
	if v2Root := strings.Index(req.URL.Path, "/v2/"); v2Root != -1 {
		pingPath = pingPath[:v2Root+4]
	} else if v1Root := strings.Index(req.URL.Path, "/v1/"); v1Root != -1 {
		pingPath = pingPath[:v1Root] + "/v2/"
	} else {
		return nil
	}

	ping := url.URL{
		Host:   req.URL.Host,
		Scheme: req.URL.Scheme,
		Path:   pingPath,
	}

	challenges, err := ea.challenges.GetChallenges(ping)
	if err != nil {
		return err
	}

	if len(challenges) > 0 {
		for _, handler := range ea.handlers {
			for _, c := range challenges {
				if c.Scheme != handler.Scheme() {
					continue
				}
				if err := handler.AuthorizeRequest(req, c.Parameters); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// This is the minimum duration a token can last (in seconds).
// A token must not live less than 60 seconds because older versions
// of the Docker client didn't read their expiration from the token
// response and assumed 60 seconds.  So to remain compatible with
// those implementations, a token must live at least this long.
const minimumTokenLifetimeSeconds = 60

// Private interface for time used by this package to enable tests to provide their own implementation.
type clock interface {
	Now() time.Time
}

type tokenHandler struct {
	header    http.Header
	creds     CredentialStore
	transport http.RoundTripper
	clock     clock

	offlineAccess bool
	forceOAuth    bool
	clientID      string
	scopes        []Scope

	tokenLock       sync.Mutex
	tokenCache      string
	tokenExpiration time.Time
}

// Scope is a type which is serializable to a string
// using the allow scope grammar.
type Scope interface {
	String() string
}

// RepositoryScope represents a token scope for access
// to a repository.
type RepositoryScope struct {
	Repository string
	Class      string
	Actions    []string
}

// String returns the string representation of the repository
// using the scope grammar
func (rs RepositoryScope) String() string {
	repoType := "repository"
	// Keep existing format for image class to maintain backwards compatibility
	// with authorization servers which do not support the expanded grammar.
	if rs.Class != "" && rs.Class != "image" {
		repoType = fmt.Sprintf("%s(%s)", repoType, rs.Class)
	}
	return fmt.Sprintf("%s:%s:%s", repoType, rs.Repository, strings.Join(rs.Actions, ","))
}

// RegistryScope represents a token scope for access
// to resources in the registry.
type RegistryScope struct {
	Name    string
	Actions []string
}

// String returns the string representation of the user
// using the scope grammar
func (rs RegistryScope) String() string {
	return fmt.Sprintf("registry:%s:%s", rs.Name, strings.Join(rs.Actions, ","))
}

// TokenHandlerOptions is used to configure a new token handler
type TokenHandlerOptions struct {
	Transport   http.RoundTripper
	Credentials CredentialStore

	OfflineAccess bool
	ForceOAuth    bool
	ClientID      string
	Scopes        []Scope
}

// An implementation of clock for providing real time data.
type realClock struct{}

// Now implements clock
func (realClock) Now() time.Time { return time.Now() }

// NewTokenHandler creates a new AuthenicationHandler which supports
// fetching tokens from a remote token server.
func NewTokenHandler(transport http.RoundTripper, creds CredentialStore, scope string, actions ...string) AuthenticationHandler {
	// Create options...
	return NewTokenHandlerWithOptions(TokenHandlerOptions{
		Transport:   transport,
		Credentials: creds,
		Scopes: []Scope{
			RepositoryScope{
				Repository: scope,
				Actions:    actions,
			},
		},
	})
}

// NewTokenHandlerWithOptions creates a new token handler using the provided
// options structure.
func NewTokenHandlerWithOptions(options TokenHandlerOptions) AuthenticationHandler {
	handler := &tokenHandler{
		transport:     options.Transport,
		creds:         options.Credentials,
		offlineAccess: options.OfflineAccess,
		forceOAuth:    options.ForceOAuth,
		clientID:      options.ClientID,
		scopes:        options.Scopes,
		clock:         realClock{},
	}

	return handler
}

func (th *tokenHandler) client() *http.Client {
	return &http.Client{
		Transport: th.transport,
		Timeout:   15 * time.Second,
	}
}

func (th *tokenHandler) Scheme() string {
	return "bearer"
}

func (th *tokenHandler) AuthorizeRequest(req *http.Request, params map[string]string) error {
	var additionalScopes []string
	if fromParam := req.URL.Query().Get("from"); fromParam != "" {
		additionalScopes = append(additionalScopes, RepositoryScope{
			Repository: fromParam,
			Actions:    []string{"pull"},
		}.String())
	}

	token, err := th.getToken(params, additionalScopes...)
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	return nil
}

func (th *tokenHandler) getToken(params map[string]string, additionalScopes ...string) (string, error) {
	th.tokenLock.Lock()
	defer th.tokenLock.Unlock()
	scopes := make([]string, 0, len(th.scopes)+len(additionalScopes))
	for _, scope := range th.scopes {
		scopes = append(scopes, scope.String())
	}
	var addedScopes bool
	for _, scope := range additionalScopes {
		scopes = append(scopes, scope)
		addedScopes = true
	}

	now := th.clock.Now()
	if now.After(th.tokenExpiration) || addedScopes {
		token, expiration, err := th.fetchToken(params, scopes)
		if err != nil {
			return "", err
		}

		// do not update cache for added scope tokens
		if !addedScopes {
			th.tokenCache = token
			th.tokenExpiration = expiration
		}

		return token, nil
	}

	return th.tokenCache, nil
}

type postTokenResponse struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresIn    int       `json:"expires_in"`
	IssuedAt     time.Time `json:"issued_at"`
	Scope        string    `json:"scope"`
}

func (th *tokenHandler) fetchTokenWithOAuth(realm *url.URL, refreshToken, service string, scopes []string) (token string, expiration time.Time, err error) {
	form := url.Values{}
	form.Set("scope", strings.Join(scopes, " "))
	form.Set("service", service)

	clientID := th.clientID
	if clientID == "" {
		// Use default client, this is a required field
		clientID = defaultClientID
	}
	form.Set("client_id", clientID)

	if refreshToken != "" {
		form.Set("grant_type", "refresh_token")
		form.Set("refresh_token", refreshToken)
	} else if th.creds != nil {
		form.Set("grant_type", "password")
		username, password := th.creds.Basic(realm)
		form.Set("username", username)
		form.Set("password", password)

		// attempt to get a refresh token
		form.Set("access_type", "offline")
	} else {
		// refuse to do oauth without a grant type
		return "", time.Time{}, fmt.Errorf("no supported grant type")
	}

	resp, err := th.client().PostForm(realm.String(), form)
	if err != nil {
		return "", time.Time{}, err
	}
	defer resp.Body.Close()

	if !client.SuccessStatus(resp.StatusCode) {
		err := client.HandleErrorResponse(resp)
		return "", time.Time{}, err
	}

	decoder := json.NewDecoder(resp.Body)

	var tr postTokenResponse
	if err = decoder.Decode(&tr); err != nil {
		return "", time.Time{}, fmt.Errorf("unable to decode token response: %s", err)
	}

	if tr.RefreshToken != "" && tr.RefreshToken != refreshToken {
		th.creds.SetRefreshToken(realm, service, tr.RefreshToken)
	}

	if tr.ExpiresIn < minimumTokenLifetimeSeconds {
		// The default/minimum lifetime.
		tr.ExpiresIn = minimumTokenLifetimeSeconds
		logrus.Debugf("Increasing token expiration to: %d seconds", tr.ExpiresIn)
	}

	if tr.IssuedAt.IsZero() {
		// issued_at is optional in the token response.
		tr.IssuedAt = th.clock.Now().UTC()
	}

	return tr.AccessToken, tr.IssuedAt.Add(time.Duration(tr.ExpiresIn) * time.Second), nil
}

type getTokenResponse struct {
	Token        string    `json:"token"`
	AccessToken  string    `json:"access_token"`
	ExpiresIn    int       `json:"expires_in"`
	IssuedAt     time.Time `json:"issued_at"`
	RefreshToken string    `json:"refresh_token"`
}

func (th *tokenHandler) fetchTokenWithBasicAuth(realm *url.URL, service string, scopes []string) (token string, expiration time.Time, err error) {

	req, err := http.NewRequest("GET", realm.String(), nil)
	if err != nil {
		return "", time.Time{}, err
	}

	reqParams := req.URL.Query()

	if service != "" {
		reqParams.Add("service", service)
	}

	for _, scope := range scopes {
		reqParams.Add("scope", scope)
	}

	if th.offlineAccess {
		reqParams.Add("offline_token", "true")
		clientID := th.clientID
		if clientID == "" {
			clientID = defaultClientID
		}
		reqParams.Add("client_id", clientID)
	}

	if th.creds != nil {
		username, password := th.creds.Basic(realm)
		if username != "" && password != "" {
			reqParams.Add("account", username)
			req.SetBasicAuth(username, password)
		}
	}

	req.URL.RawQuery = reqParams.Encode()

	resp, err := th.client().Do(req)
	if err != nil {
		return "", time.Time{}, err
	}
	defer resp.Body.Close()

	if !client.SuccessStatus(resp.StatusCode) {
		err := client.HandleErrorResponse(resp)
		return "", time.Time{}, err
	}

	decoder := json.NewDecoder(resp.Body)

	var tr getTokenResponse
	if err = decoder.Decode(&tr); err != nil {
		return "", time.Time{}, fmt.Errorf("unable to decode token response: %s", err)
	}

	if tr.RefreshToken != "" && th.creds != nil {
		th.creds.SetRefreshToken(realm, service, tr.RefreshToken)
	}

	// `access_token` is equivalent to `token` and if both are specified
	// the choice is undefined.  Canonicalize `access_token` by sticking
	// things in `token`.
	if tr.AccessToken != "" {
		tr.Token = tr.AccessToken
	}

	if tr.Token == "" {
		return "", time.Time{}, ErrNoToken
	}

	if tr.ExpiresIn < minimumTokenLifetimeSeconds {
		// The default/minimum lifetime.
		tr.ExpiresIn = minimumTokenLifetimeSeconds
		logrus.Debugf("Increasing token expiration to: %d seconds", tr.ExpiresIn)
	}

	if tr.IssuedAt.IsZero() {
		// issued_at is optional in the token response.
		tr.IssuedAt = th.clock.Now().UTC()
	}

	return tr.Token, tr.IssuedAt.Add(time.Duration(tr.ExpiresIn) * time.Second), nil
}

func (th *tokenHandler) fetchToken(params map[string]string, scopes []string) (token string, expiration time.Time, err error) {
	realm, ok := params["realm"]
	if !ok {
		return "", time.Time{}, errors.New("no realm specified for token auth challenge")
	}

	// TODO(dmcgowan): Handle empty scheme and relative realm
	realmURL, err := url.Parse(realm)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("invalid token auth challenge realm: %s", err)
	}

	service := params["service"]

	var refreshToken string

	if th.creds != nil {
		refreshToken = th.creds.RefreshToken(realmURL, service)
	}

	if refreshToken != "" || th.forceOAuth {
		return th.fetchTokenWithOAuth(realmURL, refreshToken, service, scopes)
	}

	return th.fetchTokenWithBasicAuth(realmURL, service, scopes)
}

type basicHandler struct {
	creds CredentialStore
}

// NewBasicHandler creaters a new authentiation handler which adds
// basic authentication credentials to a request.
func NewBasicHandler(creds CredentialStore) AuthenticationHandler {
	return &basicHandler{
		creds: creds,
	}
}

func (*basicHandler) Scheme() string {
	return "basic"
}

func (bh *basicHandler) AuthorizeRequest(req *http.Request, params map[string]string) error {
	if bh.creds != nil {
		username, password := bh.creds.Basic(req.URL)
		if username != "" && password != "" {
			req.SetBasicAuth(username, password)
			return nil
		}
	}
	return ErrNoBasicAuthCredentials
}
