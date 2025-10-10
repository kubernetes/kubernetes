// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stsexchange

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"golang.org/x/oauth2"
)

func defaultHeader() http.Header {
	header := make(http.Header)
	header.Add("Content-Type", "application/x-www-form-urlencoded")
	return header
}

// ExchangeToken performs an oauth2 token exchange with the provided endpoint.
// The first 4 fields are all mandatory.  headers can be used to pass additional
// headers beyond the bare minimum required by the token exchange.  options can
// be used to pass additional JSON-structured options to the remote server.
func ExchangeToken(ctx context.Context, endpoint string, request *TokenExchangeRequest, authentication ClientAuthentication, headers http.Header, options map[string]any) (*Response, error) {
	data := url.Values{}
	data.Set("audience", request.Audience)
	data.Set("grant_type", "urn:ietf:params:oauth:grant-type:token-exchange")
	data.Set("requested_token_type", "urn:ietf:params:oauth:token-type:access_token")
	data.Set("subject_token_type", request.SubjectTokenType)
	data.Set("subject_token", request.SubjectToken)
	data.Set("scope", strings.Join(request.Scope, " "))
	if options != nil {
		opts, err := json.Marshal(options)
		if err != nil {
			return nil, fmt.Errorf("oauth2/google: failed to marshal additional options: %v", err)
		}
		data.Set("options", string(opts))
	}

	return makeRequest(ctx, endpoint, data, authentication, headers)
}

func RefreshAccessToken(ctx context.Context, endpoint string, refreshToken string, authentication ClientAuthentication, headers http.Header) (*Response, error) {
	data := url.Values{}
	data.Set("grant_type", "refresh_token")
	data.Set("refresh_token", refreshToken)

	return makeRequest(ctx, endpoint, data, authentication, headers)
}

func makeRequest(ctx context.Context, endpoint string, data url.Values, authentication ClientAuthentication, headers http.Header) (*Response, error) {
	if headers == nil {
		headers = defaultHeader()
	}
	client := oauth2.NewClient(ctx, nil)
	authentication.InjectAuthentication(data, headers)
	encodedData := data.Encode()

	req, err := http.NewRequest("POST", endpoint, strings.NewReader(encodedData))
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: failed to properly build http request: %v", err)
	}
	req = req.WithContext(ctx)
	for key, list := range headers {
		for _, val := range list {
			req.Header.Add(key, val)
		}
	}
	req.Header.Add("Content-Length", strconv.Itoa(len(encodedData)))

	resp, err := client.Do(req)

	if err != nil {
		return nil, fmt.Errorf("oauth2/google: invalid response from Secure Token Server: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return nil, fmt.Errorf("oauth2/google: status code %d: %s", c, body)
	}
	var stsResp Response
	err = json.Unmarshal(body, &stsResp)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: failed to unmarshal response body from Secure Token Server: %v", err)

	}

	return &stsResp, nil
}

// TokenExchangeRequest contains fields necessary to make an oauth2 token exchange.
type TokenExchangeRequest struct {
	ActingParty struct {
		ActorToken     string
		ActorTokenType string
	}
	GrantType          string
	Resource           string
	Audience           string
	Scope              []string
	RequestedTokenType string
	SubjectToken       string
	SubjectTokenType   string
}

// Response is used to decode the remote server response during an oauth2 token exchange.
type Response struct {
	AccessToken     string `json:"access_token"`
	IssuedTokenType string `json:"issued_token_type"`
	TokenType       string `json:"token_type"`
	ExpiresIn       int    `json:"expires_in"`
	Scope           string `json:"scope"`
	RefreshToken    string `json:"refresh_token"`
}
