/*
 *
 * Copyright 2015, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Package oauth implements gRPC credentials using OAuth.
package oauth

import (
	"fmt"
	"io/ioutil"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/jwt"
	"google.golang.org/grpc/credentials"
)

// TokenSource supplies PerRPCCredentials from an oauth2.TokenSource.
type TokenSource struct {
	oauth2.TokenSource
}

// GetRequestMetadata gets the request metadata as a map from a TokenSource.
func (ts TokenSource) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"authorization": token.Type() + " " + token.AccessToken,
	}, nil
}

// RequireTransportSecurity indicates whether the credentails requires transport security.
func (ts TokenSource) RequireTransportSecurity() bool {
	return true
}

type jwtAccess struct {
	jsonKey []byte
}

// NewJWTAccessFromFile creates PerRPCCredentials from the given keyFile.
func NewJWTAccessFromFile(keyFile string) (credentials.PerRPCCredentials, error) {
	jsonKey, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("credentials: failed to read the service account key file: %v", err)
	}
	return NewJWTAccessFromKey(jsonKey)
}

// NewJWTAccessFromKey creates PerRPCCredentials from the given jsonKey.
func NewJWTAccessFromKey(jsonKey []byte) (credentials.PerRPCCredentials, error) {
	return jwtAccess{jsonKey}, nil
}

func (j jwtAccess) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	ts, err := google.JWTAccessTokenSourceFromJSON(j.jsonKey, uri[0])
	if err != nil {
		return nil, err
	}
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"authorization": token.TokenType + " " + token.AccessToken,
	}, nil
}

func (j jwtAccess) RequireTransportSecurity() bool {
	return true
}

// oauthAccess supplies PerRPCCredentials from a given token.
type oauthAccess struct {
	token oauth2.Token
}

// NewOauthAccess constructs the PerRPCCredentials using a given token.
func NewOauthAccess(token *oauth2.Token) credentials.PerRPCCredentials {
	return oauthAccess{token: *token}
}

func (oa oauthAccess) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{
		"authorization": oa.token.TokenType + " " + oa.token.AccessToken,
	}, nil
}

func (oa oauthAccess) RequireTransportSecurity() bool {
	return true
}

// NewComputeEngine constructs the PerRPCCredentials that fetches access tokens from
// Google Compute Engine (GCE)'s metadata server. It is only valid to use this
// if your program is running on a GCE instance.
// TODO(dsymonds): Deprecate and remove this.
func NewComputeEngine() credentials.PerRPCCredentials {
	return TokenSource{google.ComputeTokenSource("")}
}

// serviceAccount represents PerRPCCredentials via JWT signing key.
type serviceAccount struct {
	config *jwt.Config
}

func (s serviceAccount) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	token, err := s.config.TokenSource(ctx).Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"authorization": token.TokenType + " " + token.AccessToken,
	}, nil
}

func (s serviceAccount) RequireTransportSecurity() bool {
	return true
}

// NewServiceAccountFromKey constructs the PerRPCCredentials using the JSON key slice
// from a Google Developers service account.
func NewServiceAccountFromKey(jsonKey []byte, scope ...string) (credentials.PerRPCCredentials, error) {
	config, err := google.JWTConfigFromJSON(jsonKey, scope...)
	if err != nil {
		return nil, err
	}
	return serviceAccount{config: config}, nil
}

// NewServiceAccountFromFile constructs the PerRPCCredentials using the JSON key file
// of a Google Developers service account.
func NewServiceAccountFromFile(keyFile string, scope ...string) (credentials.PerRPCCredentials, error) {
	jsonKey, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("credentials: failed to read the service account key file: %v", err)
	}
	return NewServiceAccountFromKey(jsonKey, scope...)
}

// NewApplicationDefault returns "Application Default Credentials". For more
// detail, see https://developers.google.com/accounts/docs/application-default-credentials.
func NewApplicationDefault(ctx context.Context, scope ...string) (credentials.PerRPCCredentials, error) {
	t, err := google.DefaultTokenSource(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return TokenSource{t}, nil
}
