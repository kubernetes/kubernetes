// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"time"

	"golang.org/x/oauth2"
)

// generateAccesstokenReq is used for service account impersonation
type generateAccessTokenReq struct {
	Delegates []string `json:"delegates,omitempty"`
	Lifetime  string   `json:"lifetime,omitempty"`
	Scope     []string `json:"scope,omitempty"`
}

type impersonateTokenResponse struct {
	AccessToken string `json:"accessToken"`
	ExpireTime  string `json:"expireTime"`
}

// ImpersonateTokenSource uses a source credential, stored in Ts, to request an access token to the provided URL.
// Scopes can be defined when the access token is requested.
type ImpersonateTokenSource struct {
	// Ctx is the execution context of the impersonation process
	// used to perform http call to the URL. Required
	Ctx context.Context
	// Ts is the source credential used to generate a token on the
	// impersonated service account. Required.
	Ts oauth2.TokenSource

	// URL is the endpoint to call to generate a token
	// on behalf the service account. Required.
	URL string
	// Scopes that the impersonated credential should have. Required.
	Scopes []string
	// Delegates are the service account email addresses in a delegation chain.
	// Each service account must be granted roles/iam.serviceAccountTokenCreator
	// on the next service account in the chain. Optional.
	Delegates []string
	// TokenLifetimeSeconds is the number of seconds the impersonation token will
	// be valid for.
	TokenLifetimeSeconds int
}

// Token performs the exchange to get a temporary service account token to allow access to GCP.
func (its ImpersonateTokenSource) Token() (*oauth2.Token, error) {
	lifetimeString := "3600s"
	if its.TokenLifetimeSeconds != 0 {
		lifetimeString = fmt.Sprintf("%ds", its.TokenLifetimeSeconds)
	}
	reqBody := generateAccessTokenReq{
		Lifetime:  lifetimeString,
		Scope:     its.Scopes,
		Delegates: its.Delegates,
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to marshal request: %v", err)
	}
	client := oauth2.NewClient(its.Ctx, its.Ts)
	req, err := http.NewRequest("POST", its.URL, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to create impersonation request: %v", err)
	}
	req = req.WithContext(its.Ctx)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to generate access token: %v", err)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to read body: %v", err)
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return nil, fmt.Errorf("oauth2/google: status code %d: %s", c, body)
	}

	var accessTokenResp impersonateTokenResponse
	if err := json.Unmarshal(body, &accessTokenResp); err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to parse response: %v", err)
	}
	expiry, err := time.Parse(time.RFC3339, accessTokenResp.ExpireTime)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: unable to parse expiry: %v", err)
	}
	return &oauth2.Token{
		AccessToken: accessTokenResp.AccessToken,
		Expiry:      expiry,
		TokenType:   "Bearer",
	}, nil
}
