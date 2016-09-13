package oauth2

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"mime"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"

	phttp "github.com/coreos/go-oidc/http"
)

// ResponseTypesEqual compares two response_type values. If either
// contains a space, it is treated as an unordered list. For example,
// comparing "code id_token" and "id_token code" would evaluate to true.
func ResponseTypesEqual(r1, r2 string) bool {
	if !strings.Contains(r1, " ") || !strings.Contains(r2, " ") {
		// fast route, no split needed
		return r1 == r2
	}

	// split, sort, and compare
	r1Fields := strings.Fields(r1)
	r2Fields := strings.Fields(r2)
	if len(r1Fields) != len(r2Fields) {
		return false
	}
	sort.Strings(r1Fields)
	sort.Strings(r2Fields)
	for i, r1Field := range r1Fields {
		if r1Field != r2Fields[i] {
			return false
		}
	}
	return true
}

const (
	// OAuth2.0 response types registered by OIDC.
	//
	// See: https://openid.net/specs/oauth-v2-multiple-response-types-1_0.html#RegistryContents
	ResponseTypeCode             = "code"
	ResponseTypeCodeIDToken      = "code id_token"
	ResponseTypeCodeIDTokenToken = "code id_token token"
	ResponseTypeIDToken          = "id_token"
	ResponseTypeIDTokenToken     = "id_token token"
	ResponseTypeToken            = "token"
	ResponseTypeNone             = "none"
)

const (
	GrantTypeAuthCode     = "authorization_code"
	GrantTypeClientCreds  = "client_credentials"
	GrantTypeUserCreds    = "password"
	GrantTypeImplicit     = "implicit"
	GrantTypeRefreshToken = "refresh_token"

	AuthMethodClientSecretPost  = "client_secret_post"
	AuthMethodClientSecretBasic = "client_secret_basic"
	AuthMethodClientSecretJWT   = "client_secret_jwt"
	AuthMethodPrivateKeyJWT     = "private_key_jwt"
)

type Config struct {
	Credentials ClientCredentials
	Scope       []string
	RedirectURL string
	AuthURL     string
	TokenURL    string

	// Must be one of the AuthMethodXXX methods above. Right now, only
	// AuthMethodClientSecretPost and AuthMethodClientSecretBasic are supported.
	AuthMethod string
}

type Client struct {
	hc          phttp.Client
	creds       ClientCredentials
	scope       []string
	authURL     *url.URL
	redirectURL *url.URL
	tokenURL    *url.URL
	authMethod  string
}

type ClientCredentials struct {
	ID     string
	Secret string
}

func NewClient(hc phttp.Client, cfg Config) (c *Client, err error) {
	if len(cfg.Credentials.ID) == 0 {
		err = errors.New("missing client id")
		return
	}

	if len(cfg.Credentials.Secret) == 0 {
		err = errors.New("missing client secret")
		return
	}

	if cfg.AuthMethod == "" {
		cfg.AuthMethod = AuthMethodClientSecretBasic
	} else if cfg.AuthMethod != AuthMethodClientSecretPost && cfg.AuthMethod != AuthMethodClientSecretBasic {
		err = fmt.Errorf("auth method %q is not supported", cfg.AuthMethod)
		return
	}

	au, err := phttp.ParseNonEmptyURL(cfg.AuthURL)
	if err != nil {
		return
	}

	tu, err := phttp.ParseNonEmptyURL(cfg.TokenURL)
	if err != nil {
		return
	}

	// Allow empty redirect URL in the case where the client
	// only needs to verify a given token.
	ru, err := url.Parse(cfg.RedirectURL)
	if err != nil {
		return
	}

	c = &Client{
		creds:       cfg.Credentials,
		scope:       cfg.Scope,
		redirectURL: ru,
		authURL:     au,
		tokenURL:    tu,
		hc:          hc,
		authMethod:  cfg.AuthMethod,
	}

	return
}

// Return the embedded HTTP client
func (c *Client) HttpClient() phttp.Client {
	return c.hc
}

// Generate the url for initial redirect to oauth provider.
func (c *Client) AuthCodeURL(state, accessType, prompt string) string {
	v := c.commonURLValues()
	v.Set("state", state)
	if strings.ToLower(accessType) == "offline" {
		v.Set("access_type", "offline")
	}

	if prompt != "" {
		v.Set("prompt", prompt)
	}
	v.Set("response_type", "code")

	q := v.Encode()
	u := *c.authURL
	if u.RawQuery == "" {
		u.RawQuery = q
	} else {
		u.RawQuery += "&" + q
	}
	return u.String()
}

func (c *Client) commonURLValues() url.Values {
	return url.Values{
		"redirect_uri": {c.redirectURL.String()},
		"scope":        {strings.Join(c.scope, " ")},
		"client_id":    {c.creds.ID},
	}
}

func (c *Client) newAuthenticatedRequest(urlToken string, values url.Values) (*http.Request, error) {
	var req *http.Request
	var err error
	switch c.authMethod {
	case AuthMethodClientSecretPost:
		values.Set("client_secret", c.creds.Secret)
		req, err = http.NewRequest("POST", urlToken, strings.NewReader(values.Encode()))
		if err != nil {
			return nil, err
		}
	case AuthMethodClientSecretBasic:
		req, err = http.NewRequest("POST", urlToken, strings.NewReader(values.Encode()))
		if err != nil {
			return nil, err
		}
		encodedID := url.QueryEscape(c.creds.ID)
		encodedSecret := url.QueryEscape(c.creds.Secret)
		req.SetBasicAuth(encodedID, encodedSecret)
	default:
		panic("misconfigured client: auth method not supported")
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return req, nil

}

// ClientCredsToken posts the client id and secret to obtain a token scoped to the OAuth2 client via the "client_credentials" grant type.
// May not be supported by all OAuth2 servers.
func (c *Client) ClientCredsToken(scope []string) (result TokenResponse, err error) {
	v := url.Values{
		"scope":      {strings.Join(scope, " ")},
		"grant_type": {GrantTypeClientCreds},
	}

	req, err := c.newAuthenticatedRequest(c.tokenURL.String(), v)
	if err != nil {
		return
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	return parseTokenResponse(resp)
}

// UserCredsToken posts the username and password to obtain a token scoped to the OAuth2 client via the "password" grant_type
// May not be supported by all OAuth2 servers.
func (c *Client) UserCredsToken(username, password string) (result TokenResponse, err error) {
	v := url.Values{
		"scope":      {strings.Join(c.scope, " ")},
		"grant_type": {GrantTypeUserCreds},
		"username":   {username},
		"password":   {password},
	}

	req, err := c.newAuthenticatedRequest(c.tokenURL.String(), v)
	if err != nil {
		return
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	return parseTokenResponse(resp)
}

// RequestToken requests a token from the Token Endpoint with the specified grantType.
// If 'grantType' == GrantTypeAuthCode, then 'value' should be the authorization code.
// If 'grantType' == GrantTypeRefreshToken, then 'value' should be the refresh token.
func (c *Client) RequestToken(grantType, value string) (result TokenResponse, err error) {
	v := c.commonURLValues()

	v.Set("grant_type", grantType)
	v.Set("client_secret", c.creds.Secret)
	switch grantType {
	case GrantTypeAuthCode:
		v.Set("code", value)
	case GrantTypeRefreshToken:
		v.Set("refresh_token", value)
	default:
		err = fmt.Errorf("unsupported grant_type: %v", grantType)
		return
	}

	req, err := c.newAuthenticatedRequest(c.tokenURL.String(), v)
	if err != nil {
		return
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	return parseTokenResponse(resp)
}

func parseTokenResponse(resp *http.Response) (result TokenResponse, err error) {
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return
	}
	badStatusCode := resp.StatusCode < 200 || resp.StatusCode > 299

	contentType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil {
		return
	}

	result = TokenResponse{
		RawBody: body,
	}

	newError := func(typ, desc, state string) error {
		if typ == "" {
			return fmt.Errorf("unrecognized error %s", body)
		}
		return &Error{typ, desc, state}
	}

	if contentType == "application/x-www-form-urlencoded" || contentType == "text/plain" {
		var vals url.Values
		vals, err = url.ParseQuery(string(body))
		if err != nil {
			return
		}
		if error := vals.Get("error"); error != "" || badStatusCode {
			err = newError(error, vals.Get("error_description"), vals.Get("state"))
			return
		}
		e := vals.Get("expires_in")
		if e == "" {
			e = vals.Get("expires")
		}
		if e != "" {
			result.Expires, err = strconv.Atoi(e)
			if err != nil {
				return
			}
		}
		result.AccessToken = vals.Get("access_token")
		result.TokenType = vals.Get("token_type")
		result.IDToken = vals.Get("id_token")
		result.RefreshToken = vals.Get("refresh_token")
		result.Scope = vals.Get("scope")
	} else {
		var r struct {
			AccessToken  string      `json:"access_token"`
			TokenType    string      `json:"token_type"`
			IDToken      string      `json:"id_token"`
			RefreshToken string      `json:"refresh_token"`
			Scope        string      `json:"scope"`
			State        string      `json:"state"`
			ExpiresIn    json.Number `json:"expires_in"` // Azure AD returns string
			Expires      int         `json:"expires"`
			Error        string      `json:"error"`
			Desc         string      `json:"error_description"`
		}
		if err = json.Unmarshal(body, &r); err != nil {
			return
		}
		if r.Error != "" || badStatusCode {
			err = newError(r.Error, r.Desc, r.State)
			return
		}
		result.AccessToken = r.AccessToken
		result.TokenType = r.TokenType
		result.IDToken = r.IDToken
		result.RefreshToken = r.RefreshToken
		result.Scope = r.Scope
		if expiresIn, err := r.ExpiresIn.Int64(); err != nil {
			result.Expires = r.Expires
		} else {
			result.Expires = int(expiresIn)
		}
	}
	return
}

type TokenResponse struct {
	AccessToken  string
	TokenType    string
	Expires      int
	IDToken      string
	RefreshToken string // OPTIONAL.
	Scope        string // OPTIONAL, if identical to the scope requested by the client, otherwise, REQUIRED.
	RawBody      []byte // In case callers need some other non-standard info from the token response
}

type AuthCodeRequest struct {
	ResponseType string
	ClientID     string
	RedirectURL  *url.URL
	Scope        []string
	State        string
}

func ParseAuthCodeRequest(q url.Values) (AuthCodeRequest, error) {
	acr := AuthCodeRequest{
		ResponseType: q.Get("response_type"),
		ClientID:     q.Get("client_id"),
		State:        q.Get("state"),
		Scope:        make([]string, 0),
	}

	qs := strings.TrimSpace(q.Get("scope"))
	if qs != "" {
		acr.Scope = strings.Split(qs, " ")
	}

	err := func() error {
		if acr.ClientID == "" {
			return NewError(ErrorInvalidRequest)
		}

		redirectURL := q.Get("redirect_uri")
		if redirectURL != "" {
			ru, err := url.Parse(redirectURL)
			if err != nil {
				return NewError(ErrorInvalidRequest)
			}
			acr.RedirectURL = ru
		}

		return nil
	}()

	return acr, err
}
