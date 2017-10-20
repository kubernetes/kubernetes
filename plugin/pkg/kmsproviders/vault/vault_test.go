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
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
)

const (
	key        = "kube-secret-enc-key"
	sampleText = "abcdefghijklmnopqrstuvwxyz"

	configOneKey = `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
token: 8dad1053-4a4e-f359-2eab-d57968eb277f
`
	configTwoKey = `
keyNames: 
  - @key@
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
token: 8dad1053-4a4e-f359-2eab-d57968eb277f
`
)

func TestOneKey(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	service, err := serviceTestFactory(configOneKey, server.URL, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, error: %s ", err)
	}

	originalText := []byte(sampleText)

	cipher, err := service.Encrypt(originalText)
	if err != nil {
		t.Fatalf("fail to encrypt data with Vault, error %s", err)
	}
	if !strings.HasPrefix(cipher, key+":v1:") {
		t.Fatalf("the cipher has no correct prefix, %s", cipher)
	}

	untransformedData, err := service.Decrypt(cipher)
	if err != nil {
		t.Fatalf("fail to decrypt data with Vault, %s ", err)
	}
	if bytes.Compare(untransformedData, originalText) != 0 {
		t.Fatalf("transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
	}
}

func TestMoreThanOneKeys(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	// Create cipher when there is one key
	service, err := serviceTestFactory(configOneKey, server.URL, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, %s ", err)
	}

	originalText := []byte(sampleText)

	cipher, err := service.Encrypt(originalText)
	if err != nil {
		t.Fatalf("fail to encrypt data with Vault, %s ", err)
	}

	// Now there are 2 keys in the service
	newKey := "new-" + key

	newService, err := serviceTestFactory(configTwoKey, server.URL, newKey, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, %s ", err)
	}

	newCipher, err := newService.Encrypt(originalText)
	if err != nil {
		t.Fatalf("fail to encrypt data with Vault, %s ", err)
	}
	// New cipher should be prefixed with new key
	if !strings.HasPrefix(newCipher, newKey+":v1:") {
		t.Errorf("the cipher has no correct prefix, %s", cipher)
	}

	// Both old cipher and new cipher should be decrypted correctly
	for _, cipherData := range []string{cipher, newCipher} {
		untransformedData, err := newService.Decrypt(cipherData)
		if err != nil {
			t.Fatalf("fail to decrypt data with Vault, %s ", err)
		}
		if !bytes.Equal(untransformedData, originalText) {
			t.Errorf("transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
		}
	}
}

func TestWithoutMatchKey(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	service, err := serviceTestFactory(configOneKey, server.URL, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, error: %s ", err)
	}

	cipher, err := service.Encrypt([]byte(sampleText))
	if err != nil {
		t.Fatalf("fail to encrypt data with Vault, error: %s ", err)
	}

	// Create a service with only new key
	newKey := "new-" + key
	newService, err := serviceTestFactory(configOneKey, server.URL, newKey)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, error: %s ", err)
	}

	_, err = newService.Decrypt(cipher)
	if err == nil {
		t.Fatal("should fail to decrypt cipher that has no match key")
	}
}

func TestWithRefreshToken(t *testing.T) {
	forbidden := true
	encryptCount := 0
	decryptCount := 0
	server := onceUseTokenServer(t, &forbidden, &encryptCount, &decryptCount)
	defer server.Close()

	// For token auth, not refresh token, so the operation will fail.
	// Also no retry, the request count should be one.
	service, err := serviceTestFactory(configOneKey, server.URL, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, error: %s ", err)
	}

	originalText := []byte(sampleText)

	cipher, err := service.Encrypt(originalText)
	if err == nil {
		t.Fatal("should be forbidden to encrypt data with Vault")
	}
	if encryptCount != 1 {
		t.Errorf("expect call encrypt 1 time, but %d times", encryptCount)
	}

	_, err = service.Decrypt(cipher)
	if err == nil {
		t.Error("should be forbidden to decrypt data with Vault")
	}
	if encryptCount != 1 {
		t.Errorf("expect call decrypt 1 time, but %d times", decryptCount)
	}

	// For approle auth, it will refresh token and retry request.
	configRole := strings.Replace(configOneKey, "token", "roleID", 1)
	service, err = serviceTestFactory(configRole, server.URL, key)
	if err != nil {
		t.Fatalf("fail to initialize Vault envelope service, %s", err)
	}

	forbidden = true
	encryptCount = 0
	decryptCount = 0

	cipher, err = service.Encrypt(originalText)
	if err != nil {
		t.Fatalf("fail to encrypt with error %s", err)
	}
	if encryptCount != 2 {
		t.Errorf("expect call encrypt 2 times, but %d times", encryptCount)
	}

	untransformedData, err := service.Decrypt(cipher)
	if err != nil {
		t.Fatalf("fail to decrypt with error %s", err)
	}
	if encryptCount != 2 {
		t.Errorf("expect call decrypt 2 times, but %d times", decryptCount)
	}
	if bytes.Compare(untransformedData, originalText) != 0 {
		t.Fatalf("transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
	}
}

func onceUseTokenServer(t *testing.T, forbidden *bool, encryptCount, decryptCount *int) *httptest.Server {
	roleLogin := approleLoginHandler{t}
	appRoleLoginFunc := func(w http.ResponseWriter, r *http.Request) {
		*forbidden = false
		roleLogin.ServeHTTP(w, r)
	}

	encrypt := encryptHandler{t}
	encryptFunc := func(w http.ResponseWriter, r *http.Request) {
		*encryptCount++
		if !*forbidden {
			encrypt.ServeHTTP(w, r)
			*forbidden = true
		} else {
			http.Error(w, http.StatusText(http.StatusForbidden), http.StatusForbidden)
		}
	}

	decrypt := decryptHandler{t}
	decryptFunc := func(w http.ResponseWriter, r *http.Request) {
		*decryptCount++
		if !*forbidden {
			decrypt.ServeHTTP(w, r)
			*forbidden = true
		} else {
			http.Error(w, http.StatusText(http.StatusForbidden), http.StatusForbidden)
		}
	}

	handlers := map[string]http.Handler{
		"/v1/auth/approle/login": http.HandlerFunc(appRoleLoginFunc),
		"/v1/transit/encrypt/":   http.HandlerFunc(encryptFunc),
		"/v1/transit/decrypt/":   http.HandlerFunc(decryptFunc),
	}

	return VaultTestServer(t, handlers)
}

func TestInvalidConfiguration(t *testing.T) {
	server := VaultTestServer(t, nil)
	defer server.Close()

	// No key name
	configWithoutKey := `
addr: @url@
vaultCACert: testdata/ca.crt
token: 8dad1053-4a4e-f359-2eab-d57968eb277f
`
	// No address
	configWithoutAddress := `
keyNames: 
  - @key@
vaultCACert: testdata/ca.crt
token: 8dad1053-4a4e-f359-2eab-d57968eb277f
`
	// No any authentication info
	configWithoutAuth := `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
`
	// tls authentication, but no client key
	configTLSWithoutClientKey := `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
clientCert: testdata/client.crt
`
	// tls authentication, but no client cert
	configTLSWithoutClientCert := `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
clientKey: testdata/client.key
`
	// approle authentication, but no role id
	configRoleWithoutRoleID := `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
secretID: cd834818-ac2b-4db0-b3e9-2cdd6db599f0
`
	// there are more than one authentication
	configMoreThanOneAuth := `
keyNames: 
  - @key@
addr: @url@
vaultCACert: testdata/ca.crt
token: 8dad1053-4a4e-f359-2eab-d57968eb277f
roleID: 655a9287-f1be-4be0-844c-4f13a1757532
`

	invalidConfigs := []struct {
		name        string
		config      string
		description string
	}{
		{"configWithoutKey", configWithoutKey, "there is no key name"},
		{"configWithoutAddress", configWithoutAddress, "there is no address"},
		{"configWithoutAuth", configWithoutAuth, "there is no authentication"},
		{"configTLSWithoutClientKey", configTLSWithoutClientKey, "there is no client key file"},
		{"configTLSWithoutClientCert", configTLSWithoutClientCert, "there is no client cert file"},
		{"configRoleWithoutRoleID", configRoleWithoutRoleID, "there is no role id"},
		{"configMoreThanOneAuth", configMoreThanOneAuth, "there are more than one authentications"},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := serviceTestFactory(testCase.config, server.URL, key)
			if err == nil {
				t.Fatal("should fail to create vault KMS service when " + testCase.description)
			}
		})
	}
}

func serviceTestFactory(config, url string, keys ...string) (envelope.Service, error) {
	config = strings.Replace(config, "@url@", url, 1)
	for _, k := range keys {
		config = strings.Replace(config, "@key@", k, 1)
	}
	return KMSFactory(strings.NewReader(config))
}
