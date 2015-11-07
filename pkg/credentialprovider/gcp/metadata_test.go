/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package gcp_credentials

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/credentialprovider"
)

func TestDockerKeyringFromGoogleDockerConfigMetadata(t *testing.T) {
	registryUrl := "hello.kubernetes.io"
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, registryUrl, email, auth)

	const probeEndpoint = "/computeMetadata/v1/"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the one metadata key.
		if probeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
		} else if strings.HasSuffix(dockerConfigKey, r.URL.Path) {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, sampleDockerConfig)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	keyring := &credentialprovider.BasicDockerKeyring{}
	provider := &dockerConfigKeyProvider{
		metadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}

	keyring.Add(provider.Provide())

	creds, ok := keyring.Lookup(registryUrl)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", registryUrl)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if username != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestDockerKeyringFromGoogleDockerConfigMetadataUrl(t *testing.T) {
	registryUrl := "hello.kubernetes.io"
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, registryUrl, email, auth)

	const probeEndpoint = "/computeMetadata/v1/"
	const valueEndpoint = "/my/value"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if probeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
		} else if valueEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, sampleDockerConfig)
		} else if strings.HasSuffix(dockerConfigUrlKey, r.URL.Path) {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/text")
			fmt.Fprint(w, "http://foo.bar.com"+valueEndpoint)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	keyring := &credentialprovider.BasicDockerKeyring{}
	provider := &dockerConfigUrlKeyProvider{
		metadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}

	keyring.Add(provider.Provide())

	creds, ok := keyring.Lookup(registryUrl)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", registryUrl)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if username != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestContainerRegistryBasics(t *testing.T) {
	registryUrl := "container.cloud.google.com"
	email := "1234@project.gserviceaccount.com"
	token := &tokenBlob{AccessToken: "ya26.lots-of-indiscernible-garbage"}

	const (
		defaultEndpoint = "/computeMetadata/v1/instance/service-accounts/default/"
		scopeEndpoint   = defaultEndpoint + "scopes"
		emailEndpoint   = defaultEndpoint + "email"
		tokenEndpoint   = defaultEndpoint + "token"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if scopeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `["%s.read_write"]`, storageScopePrefix)
		} else if emailEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, email)
		} else if tokenEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			bytes, err := json.Marshal(token)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			fmt.Fprintln(w, string(bytes))
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	keyring := &credentialprovider.BasicDockerKeyring{}
	provider := &containerRegistryProvider{
		metadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}

	keyring.Add(provider.Provide())

	creds, ok := keyring.Lookup(registryUrl)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", registryUrl)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if "_token" != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", "_token", val.Username)
	}
	if token.AccessToken != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", token.AccessToken, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestContainerRegistryNoStorageScope(t *testing.T) {
	const (
		defaultEndpoint = "/computeMetadata/v1/instance/service-accounts/default/"
		scopeEndpoint   = defaultEndpoint + "scopes"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if scopeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `["https://www.googleapis.com/auth/compute.read_write"]`)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	provider := &containerRegistryProvider{
		metadataProvider{Client: &http.Client{Transport: transport}},
	}

	if provider.Enabled() {
		t.Errorf("Provider is unexpectedly enabled")
	}
}

func TestComputePlatformScopeSubstitutesStorageScope(t *testing.T) {
	const (
		defaultEndpoint = "/computeMetadata/v1/instance/service-accounts/default/"
		scopeEndpoint   = defaultEndpoint + "scopes"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if scopeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `["https://www.googleapis.com/auth/compute.read_write","https://www.googleapis.com/auth/cloud-platform.read-only"]`)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	provider := &containerRegistryProvider{
		metadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}
}

func TestAllProvidersNoMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := &http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	}

	providers := []credentialprovider.DockerConfigProvider{
		&dockerConfigKeyProvider{
			metadataProvider{Client: &http.Client{Transport: transport}},
		},
		&dockerConfigUrlKeyProvider{
			metadataProvider{Client: &http.Client{Transport: transport}},
		},
		&containerRegistryProvider{
			metadataProvider{Client: &http.Client{Transport: transport}},
		},
	}

	for _, provider := range providers {
		if provider.Enabled() {
			t.Errorf("Provider %s is unexpectedly enabled", reflect.TypeOf(provider).String())
		}
	}
}
