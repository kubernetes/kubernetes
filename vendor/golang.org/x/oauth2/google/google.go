// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"cloud.google.com/go/compute/metadata"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google/externalaccount"
	"golang.org/x/oauth2/google/internal/externalaccountauthorizeduser"
	"golang.org/x/oauth2/google/internal/impersonate"
	"golang.org/x/oauth2/jwt"
)

// Endpoint is Google's OAuth 2.0 default endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:       "https://accounts.google.com/o/oauth2/auth",
	TokenURL:      "https://oauth2.googleapis.com/token",
	DeviceAuthURL: "https://oauth2.googleapis.com/device/code",
	AuthStyle:     oauth2.AuthStyleInParams,
}

// MTLSTokenURL is Google's OAuth 2.0 default mTLS endpoint.
const MTLSTokenURL = "https://oauth2.mtls.googleapis.com/token"

// JWTTokenURL is Google's OAuth 2.0 token URL to use with the JWT flow.
const JWTTokenURL = "https://oauth2.googleapis.com/token"

// ConfigFromJSON uses a Google Developers Console client_credentials.json
// file to construct a config.
// client_credentials.json can be downloaded from
// https://console.developers.google.com, under "Credentials". Download the Web
// application credentials in the JSON format and provide the contents of the
// file as jsonKey.
func ConfigFromJSON(jsonKey []byte, scope ...string) (*oauth2.Config, error) {
	type cred struct {
		ClientID     string   `json:"client_id"`
		ClientSecret string   `json:"client_secret"`
		RedirectURIs []string `json:"redirect_uris"`
		AuthURI      string   `json:"auth_uri"`
		TokenURI     string   `json:"token_uri"`
	}
	var j struct {
		Web       *cred `json:"web"`
		Installed *cred `json:"installed"`
	}
	if err := json.Unmarshal(jsonKey, &j); err != nil {
		return nil, err
	}
	var c *cred
	switch {
	case j.Web != nil:
		c = j.Web
	case j.Installed != nil:
		c = j.Installed
	default:
		return nil, fmt.Errorf("oauth2/google: no credentials found")
	}
	if len(c.RedirectURIs) < 1 {
		return nil, errors.New("oauth2/google: missing redirect URL in the client_credentials.json")
	}
	return &oauth2.Config{
		ClientID:     c.ClientID,
		ClientSecret: c.ClientSecret,
		RedirectURL:  c.RedirectURIs[0],
		Scopes:       scope,
		Endpoint: oauth2.Endpoint{
			AuthURL:  c.AuthURI,
			TokenURL: c.TokenURI,
		},
	}, nil
}

// JWTConfigFromJSON uses a Google Developers service account JSON key file to read
// the credentials that authorize and authenticate the requests.
// Create a service account on "Credentials" for your project at
// https://console.developers.google.com to download a JSON key file.
func JWTConfigFromJSON(jsonKey []byte, scope ...string) (*jwt.Config, error) {
	var f credentialsFile
	if err := json.Unmarshal(jsonKey, &f); err != nil {
		return nil, err
	}
	if f.Type != serviceAccountKey {
		return nil, fmt.Errorf("google: read JWT from JSON credentials: 'type' field is %q (expected %q)", f.Type, serviceAccountKey)
	}
	scope = append([]string(nil), scope...) // copy
	return f.jwtConfig(scope, ""), nil
}

// JSON key file types.
const (
	serviceAccountKey                = "service_account"
	userCredentialsKey               = "authorized_user"
	externalAccountKey               = "external_account"
	externalAccountAuthorizedUserKey = "external_account_authorized_user"
	impersonatedServiceAccount       = "impersonated_service_account"
)

// credentialsFile is the unmarshalled representation of a credentials file.
type credentialsFile struct {
	Type string `json:"type"`

	// Service Account fields
	ClientEmail    string `json:"client_email"`
	PrivateKeyID   string `json:"private_key_id"`
	PrivateKey     string `json:"private_key"`
	AuthURL        string `json:"auth_uri"`
	TokenURL       string `json:"token_uri"`
	ProjectID      string `json:"project_id"`
	UniverseDomain string `json:"universe_domain"`

	// User Credential fields
	// (These typically come from gcloud auth.)
	ClientSecret string `json:"client_secret"`
	ClientID     string `json:"client_id"`
	RefreshToken string `json:"refresh_token"`

	// External Account fields
	Audience                       string                           `json:"audience"`
	SubjectTokenType               string                           `json:"subject_token_type"`
	TokenURLExternal               string                           `json:"token_url"`
	TokenInfoURL                   string                           `json:"token_info_url"`
	ServiceAccountImpersonationURL string                           `json:"service_account_impersonation_url"`
	ServiceAccountImpersonation    serviceAccountImpersonationInfo  `json:"service_account_impersonation"`
	Delegates                      []string                         `json:"delegates"`
	CredentialSource               externalaccount.CredentialSource `json:"credential_source"`
	QuotaProjectID                 string                           `json:"quota_project_id"`
	WorkforcePoolUserProject       string                           `json:"workforce_pool_user_project"`

	// External Account Authorized User fields
	RevokeURL string `json:"revoke_url"`

	// Service account impersonation
	SourceCredentials *credentialsFile `json:"source_credentials"`
}

type serviceAccountImpersonationInfo struct {
	TokenLifetimeSeconds int `json:"token_lifetime_seconds"`
}

func (f *credentialsFile) jwtConfig(scopes []string, subject string) *jwt.Config {
	cfg := &jwt.Config{
		Email:        f.ClientEmail,
		PrivateKey:   []byte(f.PrivateKey),
		PrivateKeyID: f.PrivateKeyID,
		Scopes:       scopes,
		TokenURL:     f.TokenURL,
		Subject:      subject, // This is the user email to impersonate
		Audience:     f.Audience,
	}
	if cfg.TokenURL == "" {
		cfg.TokenURL = JWTTokenURL
	}
	return cfg
}

func (f *credentialsFile) tokenSource(ctx context.Context, params CredentialsParams) (oauth2.TokenSource, error) {
	switch f.Type {
	case serviceAccountKey:
		cfg := f.jwtConfig(params.Scopes, params.Subject)
		return cfg.TokenSource(ctx), nil
	case userCredentialsKey:
		cfg := &oauth2.Config{
			ClientID:     f.ClientID,
			ClientSecret: f.ClientSecret,
			Scopes:       params.Scopes,
			Endpoint: oauth2.Endpoint{
				AuthURL:   f.AuthURL,
				TokenURL:  f.TokenURL,
				AuthStyle: oauth2.AuthStyleInParams,
			},
		}
		if cfg.Endpoint.AuthURL == "" {
			cfg.Endpoint.AuthURL = Endpoint.AuthURL
		}
		if cfg.Endpoint.TokenURL == "" {
			if params.TokenURL != "" {
				cfg.Endpoint.TokenURL = params.TokenURL
			} else {
				cfg.Endpoint.TokenURL = Endpoint.TokenURL
			}
		}
		tok := &oauth2.Token{RefreshToken: f.RefreshToken}
		return cfg.TokenSource(ctx, tok), nil
	case externalAccountKey:
		cfg := &externalaccount.Config{
			Audience:                       f.Audience,
			SubjectTokenType:               f.SubjectTokenType,
			TokenURL:                       f.TokenURLExternal,
			TokenInfoURL:                   f.TokenInfoURL,
			ServiceAccountImpersonationURL: f.ServiceAccountImpersonationURL,
			ServiceAccountImpersonationLifetimeSeconds: f.ServiceAccountImpersonation.TokenLifetimeSeconds,
			ClientSecret:             f.ClientSecret,
			ClientID:                 f.ClientID,
			CredentialSource:         &f.CredentialSource,
			QuotaProjectID:           f.QuotaProjectID,
			Scopes:                   params.Scopes,
			WorkforcePoolUserProject: f.WorkforcePoolUserProject,
		}
		return externalaccount.NewTokenSource(ctx, *cfg)
	case externalAccountAuthorizedUserKey:
		cfg := &externalaccountauthorizeduser.Config{
			Audience:       f.Audience,
			RefreshToken:   f.RefreshToken,
			TokenURL:       f.TokenURLExternal,
			TokenInfoURL:   f.TokenInfoURL,
			ClientID:       f.ClientID,
			ClientSecret:   f.ClientSecret,
			RevokeURL:      f.RevokeURL,
			QuotaProjectID: f.QuotaProjectID,
			Scopes:         params.Scopes,
		}
		return cfg.TokenSource(ctx)
	case impersonatedServiceAccount:
		if f.ServiceAccountImpersonationURL == "" || f.SourceCredentials == nil {
			return nil, errors.New("missing 'source_credentials' field or 'service_account_impersonation_url' in credentials")
		}

		ts, err := f.SourceCredentials.tokenSource(ctx, params)
		if err != nil {
			return nil, err
		}
		imp := impersonate.ImpersonateTokenSource{
			Ctx:       ctx,
			URL:       f.ServiceAccountImpersonationURL,
			Scopes:    params.Scopes,
			Ts:        ts,
			Delegates: f.Delegates,
		}
		return oauth2.ReuseTokenSource(nil, imp), nil
	case "":
		return nil, errors.New("missing 'type' field in credentials")
	default:
		return nil, fmt.Errorf("unknown credential type: %q", f.Type)
	}
}

// ComputeTokenSource returns a token source that fetches access tokens
// from Google Compute Engine (GCE)'s metadata server. It's only valid to use
// this token source if your program is running on a GCE instance.
// If no account is specified, "default" is used.
// If no scopes are specified, a set of default scopes are automatically granted.
// Further information about retrieving access tokens from the GCE metadata
// server can be found at https://cloud.google.com/compute/docs/authentication.
func ComputeTokenSource(account string, scope ...string) oauth2.TokenSource {
	return computeTokenSource(account, 0, scope...)
}

func computeTokenSource(account string, earlyExpiry time.Duration, scope ...string) oauth2.TokenSource {
	return oauth2.ReuseTokenSourceWithExpiry(nil, computeSource{account: account, scopes: scope}, earlyExpiry)
}

type computeSource struct {
	account string
	scopes  []string
}

func (cs computeSource) Token() (*oauth2.Token, error) {
	if !metadata.OnGCE() {
		return nil, errors.New("oauth2/google: can't get a token from the metadata service; not running on GCE")
	}
	acct := cs.account
	if acct == "" {
		acct = "default"
	}
	tokenURI := "instance/service-accounts/" + acct + "/token"
	if len(cs.scopes) > 0 {
		v := url.Values{}
		v.Set("scopes", strings.Join(cs.scopes, ","))
		tokenURI = tokenURI + "?" + v.Encode()
	}
	tokenJSON, err := metadata.Get(tokenURI)
	if err != nil {
		return nil, err
	}
	var res struct {
		AccessToken  string `json:"access_token"`
		ExpiresInSec int    `json:"expires_in"`
		TokenType    string `json:"token_type"`
	}
	err = json.NewDecoder(strings.NewReader(tokenJSON)).Decode(&res)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: invalid token JSON from metadata: %v", err)
	}
	if res.ExpiresInSec == 0 || res.AccessToken == "" {
		return nil, fmt.Errorf("oauth2/google: incomplete token received from metadata")
	}
	tok := &oauth2.Token{
		AccessToken: res.AccessToken,
		TokenType:   res.TokenType,
		Expiry:      time.Now().Add(time.Duration(res.ExpiresInSec) * time.Second),
	}
	// NOTE(cbro): add hidden metadata about where the token is from.
	// This is needed for detection by client libraries to know that credentials come from the metadata server.
	// This may be removed in a future version of this library.
	return tok.WithExtra(map[string]interface{}{
		"oauth2.google.tokenSource":    "compute-metadata",
		"oauth2.google.serviceAccount": acct,
	}), nil
}
