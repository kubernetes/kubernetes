// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package impersonate is used to impersonate Google Credentials.
package impersonate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/oauth2"
)

// Config for generating impersonated credentials.
type Config struct {
	// Target is the service account to impersonate. Required.
	Target string
	// Scopes the impersonated credential should have. Required.
	Scopes []string
	// Delegates are the service accounts in a delegation chain. Each service
	// account must be granted roles/iam.serviceAccountTokenCreator on the next
	// service account in the chain. Optional.
	Delegates []string
}

// TokenSource returns an impersonated TokenSource configured with the provided
// config using ts as the base credential provider for making requests.
func TokenSource(ctx context.Context, ts oauth2.TokenSource, config *Config) (oauth2.TokenSource, error) {
	if len(config.Scopes) == 0 {
		return nil, fmt.Errorf("impersonate: scopes must be provided")
	}
	its := impersonatedTokenSource{
		ctx:  ctx,
		ts:   ts,
		name: formatIAMServiceAccountName(config.Target),
		// Default to the longest acceptable value of one hour as the token will
		// be refreshed automatically.
		lifetime: "3600s",
	}

	its.delegates = make([]string, len(config.Delegates))
	for i, v := range config.Delegates {
		its.delegates[i] = formatIAMServiceAccountName(v)
	}
	its.scopes = make([]string, len(config.Scopes))
	copy(its.scopes, config.Scopes)

	return oauth2.ReuseTokenSource(nil, its), nil
}

func formatIAMServiceAccountName(name string) string {
	return fmt.Sprintf("projects/-/serviceAccounts/%s", name)
}

type generateAccessTokenReq struct {
	Delegates []string `json:"delegates,omitempty"`
	Lifetime  string   `json:"lifetime,omitempty"`
	Scope     []string `json:"scope,omitempty"`
}

type generateAccessTokenResp struct {
	AccessToken string `json:"accessToken"`
	ExpireTime  string `json:"expireTime"`
}

type impersonatedTokenSource struct {
	ctx context.Context
	ts  oauth2.TokenSource

	name      string
	lifetime  string
	scopes    []string
	delegates []string
}

// Token returns an impersonated Token.
func (i impersonatedTokenSource) Token() (*oauth2.Token, error) {
	hc := oauth2.NewClient(i.ctx, i.ts)
	reqBody := generateAccessTokenReq{
		Delegates: i.delegates,
		Lifetime:  i.lifetime,
		Scope:     i.scopes,
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("impersonate: unable to marshal request: %v", err)
	}
	url := fmt.Sprintf("https://iamcredentials.googleapis.com/v1/%s:generateAccessToken", i.name)
	req, err := http.NewRequest("POST", url, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("impersonate: unable to create request: %v", err)
	}
	req = req.WithContext(i.ctx)
	req.Header.Set("Content-Type", "application/json")

	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("impersonate: unable to generate access token: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("impersonate: unable to read body: %v", err)
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return nil, fmt.Errorf("impersonate: status code %d: %s", c, body)
	}

	var accessTokenResp generateAccessTokenResp
	if err := json.Unmarshal(body, &accessTokenResp); err != nil {
		return nil, fmt.Errorf("impersonate: unable to parse response: %v", err)
	}
	expiry, err := time.Parse(time.RFC3339, accessTokenResp.ExpireTime)
	if err != nil {
		return nil, fmt.Errorf("impersonate: unable to parse expiry: %v", err)
	}
	return &oauth2.Token{
		AccessToken: accessTokenResp.AccessToken,
		Expiry:      expiry,
	}, nil
}
