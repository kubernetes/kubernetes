/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package oauth implements gRPC credentials using OAuth.
package oauth

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"sync"

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
func (ts TokenSource) GetRequestMetadata(ctx context.Context, _ ...string) (map[string]string, error) {
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	ri, _ := credentials.RequestInfoFromContext(ctx)
	if err = credentials.CheckSecurityLevel(ri.AuthInfo, credentials.PrivacyAndIntegrity); err != nil {
		return nil, fmt.Errorf("unable to transfer TokenSource PerRPCCredentials: %v", err)
	}
	return map[string]string{
		"authorization": token.Type() + " " + token.AccessToken,
	}, nil
}

// RequireTransportSecurity indicates whether the credentials requires transport security.
func (ts TokenSource) RequireTransportSecurity() bool {
	return true
}

// removeServiceNameFromJWTURI removes RPC service name from URI.
func removeServiceNameFromJWTURI(uri string) (string, error) {
	parsed, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	parsed.Path = "/"
	return parsed.String(), nil
}

type jwtAccess struct {
	jsonKey []byte
}

// NewJWTAccessFromFile creates PerRPCCredentials from the given keyFile.
func NewJWTAccessFromFile(keyFile string) (credentials.PerRPCCredentials, error) {
	jsonKey, err := os.ReadFile(keyFile)
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
	// Remove RPC service name from URI that will be used as audience
	// in a self-signed JWT token. It follows https://google.aip.dev/auth/4111.
	aud, err := removeServiceNameFromJWTURI(uri[0])
	if err != nil {
		return nil, err
	}
	// TODO: the returned TokenSource is reusable. Store it in a sync.Map, with
	// uri as the key, to avoid recreating for every RPC.
	ts, err := google.JWTAccessTokenSourceFromJSON(j.jsonKey, aud)
	if err != nil {
		return nil, err
	}
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	ri, _ := credentials.RequestInfoFromContext(ctx)
	if err = credentials.CheckSecurityLevel(ri.AuthInfo, credentials.PrivacyAndIntegrity); err != nil {
		return nil, fmt.Errorf("unable to transfer jwtAccess PerRPCCredentials: %v", err)
	}
	return map[string]string{
		"authorization": token.Type() + " " + token.AccessToken,
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
//
// Deprecated: use oauth.TokenSource instead.
func NewOauthAccess(token *oauth2.Token) credentials.PerRPCCredentials {
	return oauthAccess{token: *token}
}

func (oa oauthAccess) GetRequestMetadata(ctx context.Context, _ ...string) (map[string]string, error) {
	ri, _ := credentials.RequestInfoFromContext(ctx)
	if err := credentials.CheckSecurityLevel(ri.AuthInfo, credentials.PrivacyAndIntegrity); err != nil {
		return nil, fmt.Errorf("unable to transfer oauthAccess PerRPCCredentials: %v", err)
	}
	return map[string]string{
		"authorization": oa.token.Type() + " " + oa.token.AccessToken,
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
	mu     sync.Mutex
	config *jwt.Config
	t      *oauth2.Token
}

func (s *serviceAccount) GetRequestMetadata(ctx context.Context, _ ...string) (map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.t.Valid() {
		var err error
		s.t, err = s.config.TokenSource(ctx).Token()
		if err != nil {
			return nil, err
		}
	}
	ri, _ := credentials.RequestInfoFromContext(ctx)
	if err := credentials.CheckSecurityLevel(ri.AuthInfo, credentials.PrivacyAndIntegrity); err != nil {
		return nil, fmt.Errorf("unable to transfer serviceAccount PerRPCCredentials: %v", err)
	}
	return map[string]string{
		"authorization": s.t.Type() + " " + s.t.AccessToken,
	}, nil
}

func (s *serviceAccount) RequireTransportSecurity() bool {
	return true
}

// NewServiceAccountFromKey constructs the PerRPCCredentials using the JSON key slice
// from a Google Developers service account.
func NewServiceAccountFromKey(jsonKey []byte, scope ...string) (credentials.PerRPCCredentials, error) {
	config, err := google.JWTConfigFromJSON(jsonKey, scope...)
	if err != nil {
		return nil, err
	}
	return &serviceAccount{config: config}, nil
}

// NewServiceAccountFromFile constructs the PerRPCCredentials using the JSON key file
// of a Google Developers service account.
func NewServiceAccountFromFile(keyFile string, scope ...string) (credentials.PerRPCCredentials, error) {
	jsonKey, err := os.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("credentials: failed to read the service account key file: %v", err)
	}
	return NewServiceAccountFromKey(jsonKey, scope...)
}

// NewApplicationDefault returns "Application Default Credentials". For more
// detail, see https://developers.google.com/accounts/docs/application-default-credentials.
func NewApplicationDefault(ctx context.Context, scope ...string) (credentials.PerRPCCredentials, error) {
	creds, err := google.FindDefaultCredentials(ctx, scope...)
	if err != nil {
		return nil, err
	}

	// If JSON is nil, the authentication is provided by the environment and not
	// with a credentials file, e.g. when code is running on Google Cloud
	// Platform. Use the returned token source.
	if creds.JSON == nil {
		return TokenSource{creds.TokenSource}, nil
	}

	// If auth is provided by env variable or creds file, the behavior will be
	// different based on whether scope is set. Because the returned
	// creds.TokenSource does oauth with jwt by default, and it requires scope.
	// We can only use it if scope is not empty, otherwise it will fail with
	// missing scope error.
	//
	// If scope is set, use it, it should just work.
	//
	// If scope is not set, we try to use jwt directly without oauth (this only
	// works if it's a service account).

	if len(scope) != 0 {
		return TokenSource{creds.TokenSource}, nil
	}

	// Try to convert JSON to a jwt config without setting the optional scope
	// parameter to check if it's a service account (the function errors if it's
	// not). This is necessary because the returned config doesn't show the type
	// of the account.
	if _, err := google.JWTConfigFromJSON(creds.JSON); err != nil {
		// If this fails, it's not a service account, return the original
		// TokenSource from above.
		return TokenSource{creds.TokenSource}, nil
	}

	// If it's a service account, create a JWT only access with the key.
	return NewJWTAccessFromKey(creds.JSON)
}
