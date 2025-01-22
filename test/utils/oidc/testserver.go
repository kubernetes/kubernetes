/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oidc

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/go-jose/go-jose.v2"
	"k8s.io/kubernetes/test/utils/oidc/handlers"
)

const (
	openIDWellKnownWebPath = "/.well-known/openid-configuration"
	authWebPath            = "/auth"
	tokenWebPath           = "/token"
	jwksWebPath            = "/jwks"
)

var (
	ErrRefreshTokenExpired = errors.New("refresh token is expired")
	ErrBadClientID         = errors.New("client ID is bad")
)

type TestServer struct {
	httpServer   *httptest.Server
	tokenHandler *handlers.MockTokenHandler
	jwksHandler  *handlers.MockJWKsHandler
}

// JwksHandler is getter of JSON Web Key Sets handler
func (ts *TestServer) JwksHandler() *handlers.MockJWKsHandler {
	return ts.jwksHandler
}

// TokenHandler is getter of JWT token handler
func (ts *TestServer) TokenHandler() *handlers.MockTokenHandler {
	return ts.tokenHandler
}

// URL returns the public URL of server
func (ts *TestServer) URL() string {
	return ts.httpServer.URL
}

// TokenURL returns the public URL of JWT token endpoint
func (ts *TestServer) TokenURL() (string, error) {
	url, err := url.JoinPath(ts.httpServer.URL, tokenWebPath)
	if err != nil {
		return "", fmt.Errorf("error joining paths: %v", err)
	}

	return url, nil
}

// BuildAndRunTestServer configures OIDC TLS server and its routing
func BuildAndRunTestServer(t *testing.T, caPath, caKeyPath, issuerOverride string) *TestServer {
	t.Helper()

	certContent, err := os.ReadFile(caPath)
	require.NoError(t, err)
	keyContent, err := os.ReadFile(caKeyPath)
	require.NoError(t, err)

	cert, err := tls.X509KeyPair(certContent, keyContent)
	require.NoError(t, err)

	mux := http.NewServeMux()
	httpServer := httptest.NewUnstartedServer(mux)
	httpServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{cert},
	}
	httpServer.StartTLS()

	t.Cleanup(func() {
		httpServer.Close()
	})

	oidcServer := &TestServer{
		httpServer:   httpServer,
		tokenHandler: handlers.NewMockTokenHandler(t),
		jwksHandler:  handlers.NewMockJWKsHandler(t),
	}

	issuer := httpServer.URL
	// issuerOverride is used to override the issuer URL in the well-known configuration.
	// This is useful to validate scenarios where discovery url is different from the issuer url.
	if len(issuerOverride) > 0 {
		issuer = issuerOverride
	}

	mux.HandleFunc(openIDWellKnownWebPath, func(writer http.ResponseWriter, request *http.Request) {
		discoveryDocHandler(t, writer, httpServer.URL, issuer)
	})

	// /c/d/bar/.well-known/openid-configuration is used to validate scenarios where discovery url is different from the issuer url
	// and discovery url contains path.
	mux.HandleFunc("/c/d/bar"+openIDWellKnownWebPath, func(writer http.ResponseWriter, request *http.Request) {
		discoveryDocHandler(t, writer, httpServer.URL, issuer)
	})

	mux.HandleFunc(tokenWebPath, func(writer http.ResponseWriter, request *http.Request) {
		token, err := oidcServer.tokenHandler.Token()
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		writer.Header().Add("Content-Type", "application/json")
		writer.WriteHeader(http.StatusOK)

		err = json.NewEncoder(writer).Encode(token)
		require.NoError(t, err)
	})

	mux.HandleFunc(authWebPath, func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc(jwksWebPath, func(writer http.ResponseWriter, request *http.Request) {
		keySet := oidcServer.jwksHandler.KeySet()

		writer.Header().Add("Content-Type", "application/json")
		writer.WriteHeader(http.StatusOK)

		err := json.NewEncoder(writer).Encode(keySet)
		require.NoError(t, err)
	})

	return oidcServer
}

func discoveryDocHandler(t *testing.T, writer http.ResponseWriter, httpServerURL, issuer string) {
	authURL, err := url.JoinPath(httpServerURL + authWebPath)
	require.NoError(t, err)
	tokenURL, err := url.JoinPath(httpServerURL + tokenWebPath)
	require.NoError(t, err)
	jwksURL, err := url.JoinPath(httpServerURL + jwksWebPath)
	require.NoError(t, err)
	userInfoURL, err := url.JoinPath(httpServerURL + authWebPath)
	require.NoError(t, err)

	writer.Header().Add("Content-Type", "application/json")

	err = json.NewEncoder(writer).Encode(struct {
		Issuer      string `json:"issuer"`
		AuthURL     string `json:"authorization_endpoint"`
		TokenURL    string `json:"token_endpoint"`
		JWKSURL     string `json:"jwks_uri"`
		UserInfoURL string `json:"userinfo_endpoint"`
	}{
		Issuer:      issuer,
		AuthURL:     authURL,
		TokenURL:    tokenURL,
		JWKSURL:     jwksURL,
		UserInfoURL: userInfoURL,
	})
	require.NoError(t, err)
}

type JosePrivateKey interface {
	*rsa.PrivateKey | *ecdsa.PrivateKey
}

// TokenHandlerBehaviorReturningPredefinedJWT describes the scenario when signed JWT token is being created.
// This behavior should being applied to the MockTokenHandler.
func TokenHandlerBehaviorReturningPredefinedJWT[K JosePrivateKey](
	t *testing.T,
	privateKey K,
	claims map[string]interface{}, accessToken, refreshToken string,
) func() (handlers.Token, error) {
	t.Helper()

	return func() (handlers.Token, error) {
		signer, err := jose.NewSigner(jose.SigningKey{Algorithm: GetSignatureAlgorithm(privateKey), Key: privateKey}, nil)
		require.NoError(t, err)

		payloadJSON, err := json.Marshal(claims)
		require.NoError(t, err)

		idTokenSignature, err := signer.Sign(payloadJSON)
		require.NoError(t, err)
		idToken, err := idTokenSignature.CompactSerialize()
		require.NoError(t, err)

		return handlers.Token{
			IDToken:      idToken,
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
		}, nil
	}
}

type JosePublicKey interface {
	*rsa.PublicKey | *ecdsa.PublicKey
}

// DefaultJwksHandlerBehavior describes the scenario when JSON Web Key Set token is being returned.
// This behavior should being applied to the MockJWKsHandler.
func DefaultJwksHandlerBehavior[K JosePublicKey](t *testing.T, verificationPublicKey K) func() jose.JSONWebKeySet {
	t.Helper()

	return func() jose.JSONWebKeySet {
		key := jose.JSONWebKey{Key: verificationPublicKey, Use: "sig", Algorithm: string(GetSignatureAlgorithm(verificationPublicKey))}

		thumbprint, err := key.Thumbprint(crypto.SHA256)
		require.NoError(t, err)

		key.KeyID = hex.EncodeToString(thumbprint)
		return jose.JSONWebKeySet{
			Keys: []jose.JSONWebKey{key},
		}
	}
}

type JoseKey interface{ JosePrivateKey | JosePublicKey }

func GetSignatureAlgorithm[K JoseKey](key K) jose.SignatureAlgorithm {
	switch any(key).(type) {
	case *rsa.PrivateKey, *rsa.PublicKey:
		return jose.RS256
	case *ecdsa.PrivateKey, *ecdsa.PublicKey:
		return jose.ES256
	default:
		panic("unknown key type") // should be impossible
	}
}
