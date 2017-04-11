package oauth2

import (
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"

	phttp "github.com/coreos/go-oidc/http"
)

func TestResponseTypesEqual(t *testing.T) {
	tests := []struct {
		r1, r2 string
		want   bool
	}{
		{"code", "code", true},
		{"id_token", "code", false},
		{"code token", "token code", true},
		{"code token", "code token", true},
		{"foo", "bar code", false},
		{"code token id_token", "token id_token code", true},
		{"code token id_token", "token id_token code zoo", false},
	}

	for i, tt := range tests {
		got1 := ResponseTypesEqual(tt.r1, tt.r2)
		got2 := ResponseTypesEqual(tt.r2, tt.r1)
		if got1 != got2 {
			t.Errorf("case %d: got different answers with different orders", i)
		}
		if tt.want != got1 {
			t.Errorf("case %d: want=%t, got=%t", i, tt.want, got1)
		}
	}
}

func TestParseAuthCodeRequest(t *testing.T) {
	tests := []struct {
		query   url.Values
		wantACR AuthCodeRequest
		wantErr error
	}{
		// no redirect_uri
		{
			query: url.Values{
				"response_type": []string{"code"},
				"scope":         []string{"foo bar baz"},
				"client_id":     []string{"XXX"},
				"state":         []string{"pants"},
			},
			wantACR: AuthCodeRequest{
				ResponseType: "code",
				ClientID:     "XXX",
				Scope:        []string{"foo", "bar", "baz"},
				State:        "pants",
				RedirectURL:  nil,
			},
		},

		// with redirect_uri
		{
			query: url.Values{
				"response_type": []string{"code"},
				"redirect_uri":  []string{"https://127.0.0.1:5555/callback?foo=bar"},
				"scope":         []string{"foo bar baz"},
				"client_id":     []string{"XXX"},
				"state":         []string{"pants"},
			},
			wantACR: AuthCodeRequest{
				ResponseType: "code",
				ClientID:     "XXX",
				Scope:        []string{"foo", "bar", "baz"},
				State:        "pants",
				RedirectURL: &url.URL{
					Scheme:   "https",
					Host:     "127.0.0.1:5555",
					Path:     "/callback",
					RawQuery: "foo=bar",
				},
			},
		},

		// unsupported response_type doesn't trigger error
		{
			query: url.Values{
				"response_type": []string{"token"},
				"redirect_uri":  []string{"https://127.0.0.1:5555/callback?foo=bar"},
				"scope":         []string{"foo bar baz"},
				"client_id":     []string{"XXX"},
				"state":         []string{"pants"},
			},
			wantACR: AuthCodeRequest{
				ResponseType: "token",
				ClientID:     "XXX",
				Scope:        []string{"foo", "bar", "baz"},
				State:        "pants",
				RedirectURL: &url.URL{
					Scheme:   "https",
					Host:     "127.0.0.1:5555",
					Path:     "/callback",
					RawQuery: "foo=bar",
				},
			},
		},

		// unparseable redirect_uri
		{
			query: url.Values{
				"response_type": []string{"code"},
				"redirect_uri":  []string{":"},
				"scope":         []string{"foo bar baz"},
				"client_id":     []string{"XXX"},
				"state":         []string{"pants"},
			},
			wantACR: AuthCodeRequest{
				ResponseType: "code",
				ClientID:     "XXX",
				Scope:        []string{"foo", "bar", "baz"},
				State:        "pants",
			},
			wantErr: NewError(ErrorInvalidRequest),
		},

		// no client_id, redirect_uri not parsed
		{
			query: url.Values{
				"response_type": []string{"code"},
				"redirect_uri":  []string{"https://127.0.0.1:5555/callback?foo=bar"},
				"scope":         []string{"foo bar baz"},
				"client_id":     []string{},
				"state":         []string{"pants"},
			},
			wantACR: AuthCodeRequest{
				ResponseType: "code",
				ClientID:     "",
				Scope:        []string{"foo", "bar", "baz"},
				State:        "pants",
				RedirectURL:  nil,
			},
			wantErr: NewError(ErrorInvalidRequest),
		},
	}

	for i, tt := range tests {
		got, err := ParseAuthCodeRequest(tt.query)
		if !reflect.DeepEqual(tt.wantErr, err) {
			t.Errorf("case %d: incorrect error value: want=%q got=%q", i, tt.wantErr, err)
		}

		if !reflect.DeepEqual(tt.wantACR, got) {
			t.Errorf("case %d: incorrect AuthCodeRequest value: want=%#v got=%#v", i, tt.wantACR, got)
		}
	}
}

type fakeBadClient struct {
	Request *http.Request
	err     error
}

func (f *fakeBadClient) Do(r *http.Request) (*http.Response, error) {
	f.Request = r
	return nil, f.err
}

func TestClientCredsToken(t *testing.T) {
	hc := &fakeBadClient{nil, errors.New("error")}
	cfg := Config{
		Credentials: ClientCredentials{ID: "c#id", Secret: "c secret"},
		Scope:       []string{"foo-scope", "bar-scope"},
		TokenURL:    "http://example.com/token",
		AuthMethod:  AuthMethodClientSecretBasic,
		RedirectURL: "http://example.com/redirect",
		AuthURL:     "http://example.com/auth",
	}

	c, err := NewClient(hc, cfg)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}

	scope := []string{"openid"}
	c.ClientCredsToken(scope)
	if hc.Request == nil {
		t.Error("request is empty")
	}

	tu := hc.Request.URL.String()
	if cfg.TokenURL != tu {
		t.Errorf("wrong token url, want=%v, got=%v", cfg.TokenURL, tu)
	}

	ct := hc.Request.Header.Get("Content-Type")
	if ct != "application/x-www-form-urlencoded" {
		t.Errorf("wrong content-type, want=application/x-www-form-urlencoded, got=%v", ct)
	}

	cid, secret, ok := phttp.BasicAuth(hc.Request)
	if !ok {
		t.Error("unexpected error parsing basic auth")
	}

	if url.QueryEscape(cfg.Credentials.ID) != cid {
		t.Errorf("wrong client ID, want=%v, got=%v", cfg.Credentials.ID, cid)
	}

	if url.QueryEscape(cfg.Credentials.Secret) != secret {
		t.Errorf("wrong client secret, want=%v, got=%v", cfg.Credentials.Secret, secret)
	}

	err = hc.Request.ParseForm()
	if err != nil {
		t.Error("unexpected error parsing form")
	}

	gt := hc.Request.PostForm.Get("grant_type")
	if gt != GrantTypeClientCreds {
		t.Errorf("wrong grant_type, want=%v, got=%v", GrantTypeClientCreds, gt)
	}

	sc := strings.Split(hc.Request.PostForm.Get("scope"), " ")
	if !reflect.DeepEqual(scope, sc) {
		t.Errorf("wrong scope, want=%v, got=%v", scope, sc)
	}
}

func TestUserCredsToken(t *testing.T) {
	hc := &fakeBadClient{nil, errors.New("error")}
	cfg := Config{
		Credentials: ClientCredentials{ID: "c#id", Secret: "c secret"},
		Scope:       []string{"foo-scope", "bar-scope"},
		TokenURL:    "http://example.com/token",
		AuthMethod:  AuthMethodClientSecretBasic,
		RedirectURL: "http://example.com/redirect",
		AuthURL:     "http://example.com/auth",
	}

	c, err := NewClient(hc, cfg)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}

	c.UserCredsToken("username", "password")
	if hc.Request == nil {
		t.Error("request is empty")
	}

	tu := hc.Request.URL.String()
	if cfg.TokenURL != tu {
		t.Errorf("wrong token url, want=%v, got=%v", cfg.TokenURL, tu)
	}

	ct := hc.Request.Header.Get("Content-Type")
	if ct != "application/x-www-form-urlencoded" {
		t.Errorf("wrong content-type, want=application/x-www-form-urlencoded, got=%v", ct)
	}

	cid, secret, ok := phttp.BasicAuth(hc.Request)
	if !ok {
		t.Error("unexpected error parsing basic auth")
	}

	if url.QueryEscape(cfg.Credentials.ID) != cid {
		t.Errorf("wrong client ID, want=%v, got=%v", cfg.Credentials.ID, cid)
	}

	if url.QueryEscape(cfg.Credentials.Secret) != secret {
		t.Errorf("wrong client secret, want=%v, got=%v", cfg.Credentials.Secret, secret)
	}

	err = hc.Request.ParseForm()
	if err != nil {
		t.Error("unexpected error parsing form")
	}

	gt := hc.Request.PostForm.Get("grant_type")
	if gt != GrantTypeUserCreds {
		t.Errorf("wrong grant_type, want=%v, got=%v", GrantTypeUserCreds, gt)
	}

	sc := strings.Split(hc.Request.PostForm.Get("scope"), " ")
	if !reflect.DeepEqual(c.scope, sc) {
		t.Errorf("wrong scope, want=%v, got=%v", c.scope, sc)
	}
}

func TestNewAuthenticatedRequest(t *testing.T) {
	tests := []struct {
		authMethod string
		url        string
		values     url.Values
	}{
		{
			authMethod: AuthMethodClientSecretBasic,
			url:        "http://example.com/token",
			values:     url.Values{},
		},
		{
			authMethod: AuthMethodClientSecretPost,
			url:        "http://example.com/token",
			values:     url.Values{},
		},
	}

	for i, tt := range tests {
		cfg := Config{
			Credentials: ClientCredentials{ID: "c#id", Secret: "c secret"},
			Scope:       []string{"foo-scope", "bar-scope"},
			TokenURL:    "http://example.com/token",
			AuthURL:     "http://example.com/auth",
			RedirectURL: "http://example.com/redirect",
			AuthMethod:  tt.authMethod,
		}
		c, err := NewClient(nil, cfg)
		req, err := c.newAuthenticatedRequest(tt.url, tt.values)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		err = req.ParseForm()
		if err != nil {
			t.Errorf("case %d: want nil err, got %v", i, err)
		}

		if tt.authMethod == AuthMethodClientSecretBasic {
			cid, secret, ok := phttp.BasicAuth(req)
			if !ok {
				t.Errorf("case %d: !ok parsing Basic Auth headers", i)
				continue
			}
			if cid != url.QueryEscape(cfg.Credentials.ID) {
				t.Errorf("case %d: want CID == %q, got CID == %q", i, cfg.Credentials.ID, cid)
			}
			if secret != url.QueryEscape(cfg.Credentials.Secret) {
				t.Errorf("case %d: want secret == %q, got secret == %q", i, cfg.Credentials.Secret, secret)
			}
		} else if tt.authMethod == AuthMethodClientSecretPost {
			if req.PostFormValue("client_secret") != cfg.Credentials.Secret {
				t.Errorf("case %d: want client_secret == %q, got client_secret == %q",
					i, cfg.Credentials.Secret, req.PostFormValue("client_secret"))
			}
		}

		for k, v := range tt.values {
			if !reflect.DeepEqual(v, req.PostForm[k]) {
				t.Errorf("case %d: key:%q want==%q, got==%q", i, k, v, req.PostForm[k])
			}
		}

		if req.URL.String() != tt.url {
			t.Errorf("case %d: want URL==%q, got URL==%q", i, tt.url, req.URL.String())
		}

	}
}

func TestParseTokenResponse(t *testing.T) {
	type response struct {
		body        string
		contentType string
		statusCode  int // defaults to http.StatusOK
	}
	tests := []struct {
		resp      response
		wantResp  TokenResponse
		wantError *Error
	}{
		{
			resp: response{
				body:        "{ \"error\": \"invalid_client\", \"state\": \"foo\" }",
				contentType: "application/json",
				statusCode:  http.StatusBadRequest,
			},
			wantError: &Error{Type: "invalid_client", State: "foo"},
		},
		{
			resp: response{
				body:        "{ \"error\": \"invalid_request\", \"state\": \"bar\" }",
				contentType: "application/json",
				statusCode:  http.StatusBadRequest,
			},
			wantError: &Error{Type: "invalid_request", State: "bar"},
		},
		{
			// Actual response from bitbucket
			resp: response{
				body:        `{"error_description": "Invalid OAuth client credentials", "error": "unauthorized_client"}`,
				contentType: "application/json",
				statusCode:  http.StatusBadRequest,
			},
			wantError: &Error{Type: "unauthorized_client", Description: "Invalid OAuth client credentials"},
		},
		{
			// Actual response from github
			resp: response{
				body:        `error=incorrect_client_credentials&error_description=The+client_id+and%2For+client_secret+passed+are+incorrect.&error_uri=https%3A%2F%2Fdeveloper.github.com%2Fv3%2Foauth%2F%23incorrect-client-credentials`,
				contentType: "application/x-www-form-urlencoded; charset=utf-8",
			},
			wantError: &Error{Type: "incorrect_client_credentials", Description: "The client_id and/or client_secret passed are incorrect."},
		},
		{
			resp: response{
				body:        `{"access_token":"e72e16c7e42f292c6912e7710c838347ae178b4a", "scope":"repo,gist", "token_type":"bearer"}`,
				contentType: "application/json",
			},
			wantResp: TokenResponse{
				AccessToken: "e72e16c7e42f292c6912e7710c838347ae178b4a",
				TokenType:   "bearer",
				Scope:       "repo,gist",
			},
		},
		{
			resp: response{
				body:        `access_token=e72e16c7e42f292c6912e7710c838347ae178b4a&scope=user%2Cgist&token_type=bearer`,
				contentType: "application/x-www-form-urlencoded",
			},
			wantResp: TokenResponse{
				AccessToken: "e72e16c7e42f292c6912e7710c838347ae178b4a",
				TokenType:   "bearer",
				Scope:       "user,gist",
			},
		},
		{
			resp: response{
				body:        `{"access_token":"foo","id_token":"bar","expires_in":200,"token_type":"bearer","refresh_token":"spam"}`,
				contentType: "application/json; charset=utf-8",
			},
			wantResp: TokenResponse{
				AccessToken:  "foo",
				IDToken:      "bar",
				Expires:      200,
				TokenType:    "bearer",
				RefreshToken: "spam",
			},
		},
		{
			// Azure AD returns "expires_in" value as string
			resp: response{
				body:        `{"access_token":"foo","id_token":"bar","expires_in":"300","token_type":"bearer","refresh_token":"spam"}`,
				contentType: "application/json; charset=utf-8",
			},
			wantResp: TokenResponse{
				AccessToken:  "foo",
				IDToken:      "bar",
				Expires:      300,
				TokenType:    "bearer",
				RefreshToken: "spam",
			},
		},
		{
			resp: response{
				body:        `{"access_token":"foo","id_token":"bar","expires":200,"token_type":"bearer","refresh_token":"spam"}`,
				contentType: "application/json; charset=utf-8",
			},
			wantResp: TokenResponse{
				AccessToken:  "foo",
				IDToken:      "bar",
				Expires:      200,
				TokenType:    "bearer",
				RefreshToken: "spam",
			},
		},
		{
			resp: response{
				body:        `access_token=foo&id_token=bar&expires_in=200&token_type=bearer&refresh_token=spam`,
				contentType: "application/x-www-form-urlencoded",
			},
			wantResp: TokenResponse{
				AccessToken:  "foo",
				IDToken:      "bar",
				Expires:      200,
				TokenType:    "bearer",
				RefreshToken: "spam",
			},
		},
	}

	for i, tt := range tests {
		r := &http.Response{
			StatusCode: http.StatusOK,
			Header: http.Header{
				"Content-Type":   []string{tt.resp.contentType},
				"Content-Length": []string{strconv.Itoa(len([]byte(tt.resp.body)))},
			},
			Body:          ioutil.NopCloser(strings.NewReader(tt.resp.body)),
			ContentLength: int64(len([]byte(tt.resp.body))),
		}
		if tt.resp.statusCode != 0 {
			r.StatusCode = tt.resp.statusCode
		}

		result, err := parseTokenResponse(r)
		if err != nil {
			if tt.wantError == nil {
				t.Errorf("case %d: got error==%v", i, err)
				continue
			}
			if !reflect.DeepEqual(tt.wantError, err) {
				t.Errorf("case %d: want=%+v, got=%+v", i, tt.wantError, err)
			}
		} else {
			if tt.wantError != nil {
				t.Errorf("case %d: want error==%v, got==nil", i, tt.wantError)
				continue
			}
			// don't compare the raw body (it's really big and clogs error messages)
			result.RawBody = tt.wantResp.RawBody
			if !reflect.DeepEqual(tt.wantResp, result) {
				t.Errorf("case %d: want=%+v, got=%+v", i, tt.wantResp, result)
			}
		}
	}
}
