// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"context"
	"fmt"
	"golang.org/x/oauth2"
	"net/http"
	"strconv"
	"time"
)

// now aliases time.Now for testing
var now = func() time.Time {
	return time.Now().UTC()
}

// Config stores the configuration for fetching tokens with external credentials.
type Config struct {
	Audience                       string
	SubjectTokenType               string
	TokenURL                       string
	TokenInfoURL                   string
	ServiceAccountImpersonationURL string
	ClientSecret                   string
	ClientID                       string
	CredentialSource               CredentialSource
	QuotaProjectID                 string
	Scopes                         []string
}

// TokenSource Returns an external account TokenSource struct. This is to be called by package google to construct a google.Credentials.
func (c *Config) TokenSource(ctx context.Context) oauth2.TokenSource {
	ts := tokenSource{
		ctx:  ctx,
		conf: c,
	}
	if c.ServiceAccountImpersonationURL == "" {
		return oauth2.ReuseTokenSource(nil, ts)
	}
	scopes := c.Scopes
	ts.conf.Scopes = []string{"https://www.googleapis.com/auth/cloud-platform"}
	imp := impersonateTokenSource{
		ctx:    ctx,
		url:    c.ServiceAccountImpersonationURL,
		scopes: scopes,
		ts:     oauth2.ReuseTokenSource(nil, ts),
	}
	return oauth2.ReuseTokenSource(nil, imp)
}

// Subject token file types.
const (
	fileTypeText = "text"
	fileTypeJSON = "json"
)

type format struct {
	// Type is either "text" or "json".  When not provided "text" type is assumed.
	Type string `json:"type"`
	// SubjectTokenFieldName is only required for JSON format.  This would be "access_token" for azure.
	SubjectTokenFieldName string `json:"subject_token_field_name"`
}

// CredentialSource stores the information necessary to retrieve the credentials for the STS exchange.
type CredentialSource struct {
	File string `json:"file"`

	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`

	EnvironmentID               string `json:"environment_id"`
	RegionURL                   string `json:"region_url"`
	RegionalCredVerificationURL string `json:"regional_cred_verification_url"`
	CredVerificationURL         string `json:"cred_verification_url"`
	Format                      format `json:"format"`
}

// parse determines the type of CredentialSource needed
func (c *Config) parse(ctx context.Context) (baseCredentialSource, error) {
	if len(c.CredentialSource.EnvironmentID) > 3 && c.CredentialSource.EnvironmentID[:3] == "aws" {
		if awsVersion, err := strconv.Atoi(c.CredentialSource.EnvironmentID[3:]); err == nil {
			if awsVersion != 1 {
				return nil, fmt.Errorf("oauth2/google: aws version '%d' is not supported in the current build", awsVersion)
			}
			return awsCredentialSource{
				EnvironmentID:               c.CredentialSource.EnvironmentID,
				RegionURL:                   c.CredentialSource.RegionURL,
				RegionalCredVerificationURL: c.CredentialSource.RegionalCredVerificationURL,
				CredVerificationURL:         c.CredentialSource.URL,
				TargetResource:              c.Audience,
				ctx:                         ctx,
			}, nil
		}
	} else if c.CredentialSource.File != "" {
		return fileCredentialSource{File: c.CredentialSource.File, Format: c.CredentialSource.Format}, nil
	} else if c.CredentialSource.URL != "" {
		return urlCredentialSource{URL: c.CredentialSource.URL, Headers: c.CredentialSource.Headers, Format: c.CredentialSource.Format, ctx: ctx}, nil
	}
	return nil, fmt.Errorf("oauth2/google: unable to parse credential source")
}

type baseCredentialSource interface {
	subjectToken() (string, error)
}

// tokenSource is the source that handles external credentials.
type tokenSource struct {
	ctx  context.Context
	conf *Config
}

// Token allows tokenSource to conform to the oauth2.TokenSource interface.
func (ts tokenSource) Token() (*oauth2.Token, error) {
	conf := ts.conf

	credSource, err := conf.parse(ts.ctx)
	if err != nil {
		return nil, err
	}
	subjectToken, err := credSource.subjectToken()

	if err != nil {
		return nil, err
	}
	stsRequest := stsTokenExchangeRequest{
		GrantType:          "urn:ietf:params:oauth:grant-type:token-exchange",
		Audience:           conf.Audience,
		Scope:              conf.Scopes,
		RequestedTokenType: "urn:ietf:params:oauth:token-type:access_token",
		SubjectToken:       subjectToken,
		SubjectTokenType:   conf.SubjectTokenType,
	}
	header := make(http.Header)
	header.Add("Content-Type", "application/x-www-form-urlencoded")
	clientAuth := clientAuthentication{
		AuthStyle:    oauth2.AuthStyleInHeader,
		ClientID:     conf.ClientID,
		ClientSecret: conf.ClientSecret,
	}
	stsResp, err := exchangeToken(ts.ctx, conf.TokenURL, &stsRequest, clientAuth, header, nil)
	if err != nil {
		return nil, err
	}

	accessToken := &oauth2.Token{
		AccessToken: stsResp.AccessToken,
		TokenType:   stsResp.TokenType,
	}
	if stsResp.ExpiresIn < 0 {
		return nil, fmt.Errorf("oauth2/google: got invalid expiry from security token service")
	} else if stsResp.ExpiresIn >= 0 {
		accessToken.Expiry = now().Add(time.Duration(stsResp.ExpiresIn) * time.Second)
	}

	if stsResp.RefreshToken != "" {
		accessToken.RefreshToken = stsResp.RefreshToken
	}
	return accessToken, nil
}
