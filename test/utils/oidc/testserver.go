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
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/go-jose/go-jose.v2"
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

type Token struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

// TokenHandler serves token responses for the test OIDC server.
// Use SetHandler to configure the response for each test scenario.
type TokenHandler struct {
	handler func() (Token, error)
}

// Token returns the configured token response.
func (h *TokenHandler) Token() (Token, error) {
	if h.handler == nil {
		return Token{}, fmt.Errorf("no token handler configured")
	}
	return h.handler()
}

// SetHandler configures the function that will be called when the token
// endpoint is hit.
func (h *TokenHandler) SetHandler(fn func() (Token, error)) {
	h.handler = fn
}

// PrependHandler sets a handler that runs for the given number of calls,
// then falls back to the previously configured handler.
func (h *TokenHandler) PrependHandler(fn func() (Token, error), times int) {
	previous := h.handler
	remaining := times
	h.handler = func() (Token, error) {
		if remaining > 0 {
			remaining--
			return fn()
		}
		if previous != nil {
			return previous()
		}
		return Token{}, fmt.Errorf("no token handler configured")
	}
}

type TestServer struct {
	httpServer   *httptest.Server
	tokenHandler *TokenHandler
	publicKeys   []jose.JSONWebKey
}

// SetPublicKey computes a thumbprint-based key ID and stores the key
// so the /jwks endpoint will serve it.
func (ts *TestServer) SetPublicKey(t *testing.T, publicKey crypto.PublicKey) {
	t.Helper()
	var alg string
	switch publicKey.(type) {
	case *rsa.PublicKey:
		alg = string(jose.RS256)
	case *ecdsa.PublicKey:
		alg = string(jose.ES256)
	default:
		t.Fatalf("unsupported public key type: %T", publicKey)
	}
	key := jose.JSONWebKey{Key: publicKey, Use: "sig", Algorithm: alg}
	thumbprint, err := key.Thumbprint(crypto.SHA256)
	require.NoError(t, err)
	key.KeyID = hex.EncodeToString(thumbprint)
	ts.publicKeys = append(ts.publicKeys, key)
}

// TokenHandler returns the token handler for configuring test responses.
func (ts *TestServer) TokenHandler() *TokenHandler {
	return ts.tokenHandler
}

// URL returns the public URL of server
func (ts *TestServer) URL() string {
	return ts.httpServer.URL
}

// TokenURL returns the public URL of JWT token endpoint
func (ts *TestServer) TokenURL() string {
	return ts.httpServer.URL + tokenWebPath
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
		tokenHandler: &TokenHandler{},
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
		if err != nil {
			t.Errorf("unexpected error %v", err)
		}
	})

	mux.HandleFunc(authWebPath, func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc(jwksWebPath, func(writer http.ResponseWriter, request *http.Request) {
		keySet := jose.JSONWebKeySet{Keys: oidcServer.publicKeys}

		writer.Header().Add("Content-Type", "application/json")
		writer.WriteHeader(http.StatusOK)

		err := json.NewEncoder(writer).Encode(keySet)
		if err != nil {
			t.Errorf("unexpected error %v", err)
		}
	})

	return oidcServer
}

func discoveryDocHandler(t *testing.T, writer http.ResponseWriter, httpServerURL, issuer string) {
	authURL := httpServerURL + authWebPath
	tokenURL := httpServerURL + tokenWebPath
	jwksURL := httpServerURL + jwksWebPath
	userInfoURL := httpServerURL + authWebPath

	writer.Header().Add("Content-Type", "application/json")

	err := json.NewEncoder(writer).Encode(struct {
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

// SignToken creates a signed JWT from the given private key and claims,
// returning the compact-serialized token string.
func SignToken[K JosePrivateKey](privateKey K, claims map[string]interface{}) (string, error) {
	signer, err := jose.NewSigner(jose.SigningKey{Algorithm: GetSignatureAlgorithm(privateKey), Key: privateKey}, nil)
	if err != nil {
		return "", fmt.Errorf("creating signer: %w", err)
	}

	payloadJSON, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("marshaling claims: %w", err)
	}

	sig, err := signer.Sign(payloadJSON)
	if err != nil {
		return "", fmt.Errorf("signing payload: %w", err)
	}

	token, err := sig.CompactSerialize()
	if err != nil {
		return "", fmt.Errorf("serializing token: %w", err)
	}
	return token, nil
}

// TokenHandlerBehaviorReturningPredefinedJWT returns a handler function that
// signs the given claims into a JWT and returns it as a Token response.
func TokenHandlerBehaviorReturningPredefinedJWT[K JosePrivateKey](
	privateKey K,
	claims map[string]interface{}, accessToken, refreshToken string,
) func() (Token, error) {
	return func() (Token, error) {
		idToken, err := SignToken(privateKey, claims)
		if err != nil {
			return Token{}, fmt.Errorf("signing id token: %w", err)
		}
		return Token{
			IDToken:      idToken,
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
		}, nil
	}
}

type JosePublicKey interface {
	*rsa.PublicKey | *ecdsa.PublicKey
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

// WriteTempFile writes content to a temporary file and returns its path.
// The file is automatically cleaned up when the test completes.
func WriteTempFile(t *testing.T, content string) string {
	t.Helper()
	file, err := os.CreateTemp("", "oidc-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Remove(file.Name()); err != nil {
			t.Fatal(err)
		}
	})
	if err := os.WriteFile(file.Name(), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return file.Name()
}
