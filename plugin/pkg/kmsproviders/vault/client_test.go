/*
Copyright 2017 The Kubernetes Authors.

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

// Package vault implements envelop encryption provider based on Vault KMS
package vault

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hashicorp/vault/api"
	"github.com/pborman/uuid"
)

const (
	cafile     = "testdata/ca.crt"
	serverCert = "testdata/server.crt"
	serverKey  = "testdata/server.key"
	clientCert = "testdata/client.crt"
	clientKey  = "testdata/client.key"
)

func TestTypicalCase(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	configToken := &EnvelopeConfig{
		Token:       uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}
	configTLS := &EnvelopeConfig{
		ClientCert:  clientCert,
		ClientKey:   clientKey,
		Address:     server.URL,
		VaultCACert: cafile,
	}
	configRole := &EnvelopeConfig{
		RoleID:      uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}

	configs := []struct {
		name   string
		config *EnvelopeConfig
	}{
		{"AuthByToken", configToken},
		{"AuthByTLS", configTLS},
		{"AuthByAppRole", configRole},
	}

	for _, testCase := range configs {
		encryptAndDecrypt(t, testCase.name, testCase.config)
	}
}

func TestCustomTransitPath(t *testing.T) {
	customPath := "custom-transit"
	server := customTransitPathServer(t, customPath)
	defer server.Close()

	config := &EnvelopeConfig{
		Token:       uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}

	// Work with custom transit path
	validPathes := []string{customPath, "/" + customPath, customPath + "/", "/" + customPath + "/"}
	for _, path := range validPathes {
		config.TransitPath = path
		encryptAndDecrypt(t, path, config)
	}

	// Invalid transit path will result 404 error
	config.TransitPath = "invalid-" + customPath
	client, err := newClientWrapper(config)
	if err != nil {
		t.Fatalf("fail to initialize Vault client: %s ", err)
	}

	_, err = client.encrypt("key", "plain")
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Error("should get 404 error for non-existed transit path")
	}

	_, err = client.decrypt("key", "cipher")
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Error("should get 404 error for non-existed transit path")
	}
}

func TestCustomAuthPath(t *testing.T) {
	customAuthPath := "custom-auth"
	server := customAuthPathServer(t, customAuthPath)
	defer server.Close()

	appRoleConfig := &EnvelopeConfig{
		RoleID:      uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}

	validAuthPaths := []string{customAuthPath, "/" + customAuthPath, customAuthPath + "/", "/" + customAuthPath + "/"}
	for _, path := range validAuthPaths {
		appRoleConfig.AuthPath = path
		encryptAndDecrypt(t, "transit", appRoleConfig)
	}

	// Invalid auth path will result 404 error
	appRoleConfig.AuthPath = "invalid-" + customAuthPath
	_, err := newClientWrapper(appRoleConfig)
	if err == nil {
		t.Error("should fail to initialize Vault client")
	}

}

func encryptAndDecrypt(t *testing.T, name string, config *EnvelopeConfig) {
	client, err := newClientWrapper(config)
	if err != nil {
		t.Fatalf("name: %s, fail to initialize Vault client: %s", name, err)
	}

	key := "key"
	text := "hello"

	cipher, err := client.encrypt(key, text)
	if err != nil {
		t.Fatalf("name: %s, fail to encrypt text, error: %s", name, err)
	}
	if !strings.HasPrefix(cipher, "vault:v1:") {
		t.Errorf("name: %s, invalid cipher text: %s", name, cipher)
	}

	plain, err := client.decrypt(key, cipher)
	if err != nil {
		t.Fatalf("name: %s, fail to decrypt text, error: %s", name, err)
	}
	if text != plain {
		t.Errorf("name: %s, expect %s, but %s", name, text, plain)
	}
}

func customTransitPathServer(t *testing.T, transit string) *httptest.Server {
	handlers := DefaultTestHandlers(t)

	// Replace with custom transit path
	for _, key := range []string{"/v1/transit/encrypt/", "/v1/transit/decrypt/"} {
		newKey := strings.Replace(key, "transit", transit, 1)
		handlers[newKey] = handlers[key]
		delete(handlers, key)
	}

	return VaultTestServer(t, handlers)
}

func customAuthPathServer(t *testing.T, auth string) *httptest.Server {
	handlers := DefaultTestHandlers(t)

	// Replace with custom auth path
	for _, key := range []string{"/v1/auth/cert/login", "/v1/auth/approle/login"} {
		newKey := strings.Replace(key, "auth", auth, 1)
		handlers[newKey] = handlers[key]
		delete(handlers, key)
	}

	return VaultTestServer(t, handlers)
}

func TestForbiddenRequest(t *testing.T) {
	forbiddenFunc := func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, http.StatusText(http.StatusForbidden), http.StatusForbidden)
	}

	handlers := map[string]http.Handler{
		"/v1/transit/encrypt/": http.HandlerFunc(forbiddenFunc),
		"/v1/transit/decrypt/": http.HandlerFunc(forbiddenFunc),
	}
	server := VaultTestServer(t, handlers)
	defer server.Close()

	config := &EnvelopeConfig{
		Token:       uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}
	client, err := newClientWrapper(config)
	if err != nil {
		t.Fatalf("fail to initialize Vault client: %s", err)
	}

	_, err = client.encrypt("key", "plain")
	if err == nil {
		t.Error("error should not be nil for 403 response")
	}
	forbidden, ok := err.(*forbiddenError)
	if !ok {
		t.Error("forbidden error should be return for 403 response")
	}
	if forbidden.err == nil {
		t.Errorf("error forbidden value, %+v", forbidden)
	}

	_, err = client.decrypt("key", "cipher")
	if err == nil {
		t.Error("error should not be nil for 403 response")
	}
	forbidden, ok = err.(*forbiddenError)
	if !ok {
		t.Error("forbidden error should be return for 403 response")
	}
	if forbidden.err == nil {
		t.Errorf("error forbidden value, %+v", forbidden)
	}
}

func TestRefreshToken(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	configTLS := &EnvelopeConfig{
		ClientCert:  clientCert,
		ClientKey:   clientKey,
		Address:     server.URL,
		VaultCACert: cafile,
	}
	configRole := &EnvelopeConfig{
		RoleID:      uuid.NewRandom().String(),
		Address:     server.URL,
		VaultCACert: cafile,
	}

	configs := []struct {
		name   string
		config *EnvelopeConfig
	}{
		{"AuthByTLS", configTLS},
		{"AuthByAppRole", configRole},
	}

	for _, testCase := range configs {
		client, err := newClientWrapper(testCase.config)
		if err != nil {
			t.Fatalf("name: %s, fail to initialize Vault client: %s", testCase.name, err)
		}

		// Refresh token successfully.
		err = client.refreshToken(testCase.config)
		if err != nil {
			t.Fatalf("name: %s, unexpecte error when refresh token: %s", testCase.name, err)
		}
	}
}

func VaultTestServer(tb testing.TB, handlers map[string]http.Handler) *httptest.Server {
	mux := http.NewServeMux()
	if handlers == nil {
		handlers = DefaultTestHandlers(tb)
	}
	for path, handler := range handlers {
		mux.Handle(path, handler)
	}
	server := httptest.NewUnstartedServer(mux)

	cert, err := tls.LoadX509KeyPair(serverCert, serverKey)
	if err != nil {
		tb.Fatalf("bad server cert and keys: %s ", err)
	}
	certs := []tls.Certificate{cert}

	ca, err := ioutil.ReadFile(cafile)
	if err != nil {
		tb.Fatalf("bad ca file: %s ", err)
	}
	certPool := x509.NewCertPool()
	if !certPool.AppendCertsFromPEM(ca) {
		tb.Fatal("failed to append certificates to pool.")
	}

	server.TLS = &tls.Config{Certificates: certs, ClientAuth: tls.VerifyClientCertIfGiven, ClientCAs: certPool}
	server.StartTLS()

	return server
}

func DefaultTestHandlers(tb testing.TB) map[string]http.Handler {
	return map[string]http.Handler{
		"/v1/transit/encrypt/":   &encryptHandler{tb},
		"/v1/transit/decrypt/":   &decryptHandler{tb},
		"/v1/auth/cert/login":    &tlsLoginHandler{tb},
		"/v1/auth/approle/login": &approleLoginHandler{tb},
	}
}

type encryptHandler struct {
	tb testing.TB
}

// Just prepend "vault:v1:" prefix as encrypted text.
func (h *encryptHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	token := r.Header.Get("X-Vault-Token")
	if token == "" {
		h.tb.Fatal("unauthenticated encrypt request.")
	}

	msg, err := parseRequest(r)
	if err != nil {
		h.tb.Fatalf("error request message for encrypt request: %v", err)
	}

	plain := msg["plaintext"].(string)
	data := map[string]interface{}{
		"ciphertext": "vault:v1:" + plain,
	}
	buildResponse(w, data)
}

type decryptHandler struct {
	tb testing.TB
}

// Remove the prefix to decrypt the text.
func (h *decryptHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	token := r.Header.Get("X-Vault-Token")
	if token == "" {
		h.tb.Fatal("unauthenticated decrypt request.")
	}

	msg, err := parseRequest(r)
	if err != nil {
		h.tb.Fatalf("error request message for decrypt request: %v", err)
	}

	cipher := msg["ciphertext"].(string)
	data := map[string]interface{}{
		"plaintext": strings.TrimPrefix(cipher, "vault:v1:"),
	}
	buildResponse(w, data)
}

type tlsLoginHandler struct {
	tb testing.TB
}

// Ensure there is client certificate for tls login
func (h *tlsLoginHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if len(r.TLS.PeerCertificates) < 1 {
		h.tb.Error("the tls login doesn't contain valid client certificate.")
	}

	buildAuthResponse(w)
}

type approleLoginHandler struct {
	tb testing.TB
}

// Ensure the request contains role id.
func (h *approleLoginHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	msg, err := parseRequest(r)
	if err != nil {
		h.tb.Fatalf("error request message for approle login: %s ", err)
	}

	roleID := msg["role_id"].(string)
	if roleID == "" {
		h.tb.Error("the approle login doesn't contain valid role id.")
	}

	buildAuthResponse(w)
}

// The request message is always json message
func parseRequest(r *http.Request) (map[string]interface{}, error) {
	var msg map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&msg)
	return msg, err
}

// Response for encrypt and decrypt
func buildResponse(w http.ResponseWriter, data map[string]interface{}) {
	secret := api.Secret{
		RequestID: uuid.NewRandom().String(),
		Data:      data,
	}

	json.NewEncoder(w).Encode(&secret)
}

// Response for login request, a client token is generated.
func buildAuthResponse(w http.ResponseWriter) {
	secret := api.Secret{
		RequestID: uuid.NewRandom().String(),
		Auth:      &api.SecretAuth{ClientToken: uuid.NewRandom().String()},
	}

	json.NewEncoder(w).Encode(&secret)
}
