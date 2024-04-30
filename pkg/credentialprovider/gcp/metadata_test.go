/*
Copyright 2014 The Kubernetes Authors.

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

package gcp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/legacy-cloud-providers/gce/gcpcredential"
)

func createProductNameFile() (string, error) {
	file, err := os.CreateTemp("", "")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary test file: %v", err)
	}
	return file.Name(), os.WriteFile(file.Name(), []byte("Google"), 0600)
}

// The tests here are run in this fashion to ensure TestAllProvidersNoMetadata
// is run after the others, since that test currently relies upon the file
// referenced by gceProductNameFile being removed, which is the opposite of
// the other tests
func TestMetadata(t *testing.T) {
	// This test requires onGCEVM to return True. On Linux, this can be faked by creating a
	// Product Name File. But on Windows, onGCEVM makes the following syscall instead:
	// wmic computersystem get model
	if runtime.GOOS == "windows" && !onGCEVM() {
		t.Skip("Skipping test on Windows, not on GCE.")
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.DisableKubeletCloudCredentialProviders, false)

	var err error
	gceProductNameFile, err = createProductNameFile()
	if err != nil {
		t.Errorf("failed to create gce product name file: %v", err)
	}
	defer os.Remove(gceProductNameFile)
	t.Run("productNameDependent", func(t *testing.T) {
		t.Run("DockerKeyringFromGoogleDockerConfigMetadata",
			DockerKeyringFromGoogleDockerConfigMetadata)
		t.Run("DockerKeyringFromGoogleDockerConfigMetadataUrl",
			DockerKeyringFromGoogleDockerConfigMetadataURL)
		t.Run("ContainerRegistryNoServiceAccount",
			ContainerRegistryNoServiceAccount)
		t.Run("ContainerRegistryBasics",
			ContainerRegistryBasics)
		t.Run("ContainerRegistryNoStorageScope",
			ContainerRegistryNoStorageScope)
		t.Run("ComputePlatformScopeSubstitutesStorageScope",
			ComputePlatformScopeSubstitutesStorageScope)
	})
	// We defer os.Remove in case of an unexpected exit, but this os.Remove call
	// is the normal teardown call so AllProvidersNoMetadata executes properly
	os.Remove(gceProductNameFile)
	t.Run("AllProvidersNoMetadata",
		AllProvidersNoMetadata)
}

func DockerKeyringFromGoogleDockerConfigMetadata(t *testing.T) {
	t.Parallel()
	registryURL := "hello.kubernetes.io"
	email := "foo@bar.baz"
	username := "foo"
	password := "bar" // Fake value for testing.
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, registryURL, email, auth)
	const probeEndpoint = "/computeMetadata/v1/"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the one metadata key.
		if probeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
		} else if strings.HasSuffix(gcpcredential.DockerConfigKey, r.URL.Path) {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, sampleDockerConfig)
		} else {
			http.Error(w, "", http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	keyring := &credentialprovider.BasicDockerKeyring{}
	provider := &DockerConfigKeyProvider{
		MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}

	keyring.Add(provider.Provide(""))

	creds, ok := keyring.Lookup(registryURL)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", registryURL)
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

func DockerKeyringFromGoogleDockerConfigMetadataURL(t *testing.T) {
	t.Parallel()
	registryURL := "hello.kubernetes.io"
	email := "foo@bar.baz"
	username := "foo"
	password := "bar" // Fake value for testing.
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, registryURL, email, auth)
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
		} else if strings.HasSuffix(gcpcredential.DockerConfigURLKey, r.URL.Path) {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/text")
			fmt.Fprint(w, "http://foo.bar.com"+valueEndpoint)
		} else {
			http.Error(w, "", http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	keyring := &credentialprovider.BasicDockerKeyring{}
	provider := &DockerConfigURLKeyProvider{
		MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}

	keyring.Add(provider.Provide(""))

	creds, ok := keyring.Lookup(registryURL)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", registryURL)
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

func ContainerRegistryBasics(t *testing.T) {
	t.Parallel()
	registryURLs := []string{"container.cloud.google.com", "eu.gcr.io", "us-west2-docker.pkg.dev"}
	for _, registryURL := range registryURLs {
		t.Run(registryURL, func(t *testing.T) {
			email := "1234@project.gserviceaccount.com"
			token := &gcpcredential.TokenBlob{AccessToken: "ya26.lots-of-indiscernible-garbage"} // Fake value for testing.

			const (
				serviceAccountsEndpoint = "/computeMetadata/v1/instance/service-accounts/"
				defaultEndpoint         = "/computeMetadata/v1/instance/service-accounts/default/"
				scopeEndpoint           = defaultEndpoint + "scopes"
				emailEndpoint           = defaultEndpoint + "email"
				tokenEndpoint           = defaultEndpoint + "token"
			)

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Only serve the URL key and the value endpoint
				if scopeEndpoint == r.URL.Path {
					w.WriteHeader(http.StatusOK)
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `["%s.read_write"]`, gcpcredential.StorageScopePrefix)
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
				} else if serviceAccountsEndpoint == r.URL.Path {
					w.WriteHeader(http.StatusOK)
					fmt.Fprintln(w, "default/\ncustom")
				} else {
					http.Error(w, "", http.StatusNotFound)
				}
			}))
			defer server.Close()

			// Make a transport that reroutes all traffic to the example server
			transport := utilnet.SetTransportDefaults(&http.Transport{
				Proxy: func(req *http.Request) (*url.URL, error) {
					return url.Parse(server.URL + req.URL.Path)
				},
			})

			keyring := &credentialprovider.BasicDockerKeyring{}
			provider := &ContainerRegistryProvider{
				MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
			}

			if !provider.Enabled() {
				t.Errorf("Provider is unexpectedly disabled")
			}

			keyring.Add(provider.Provide(""))

			creds, ok := keyring.Lookup(registryURL)
			if !ok {
				t.Errorf("Didn't find expected URL: %s", registryURL)
				return
			}
			if len(creds) > 1 {
				t.Errorf("Got more hits than expected: %s", creds)
			}
			val := creds[0]

			if val.Username != "_token" {
				t.Errorf("Unexpected username value, want: %s, got: %s", "_token", val.Username)
			}
			if token.AccessToken != val.Password {
				t.Errorf("Unexpected password value, want: %s, got: %s", token.AccessToken, val.Password)
			}
			if email != val.Email {
				t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
			}
		})
	}
}

func ContainerRegistryNoServiceAccount(t *testing.T) {
	const (
		serviceAccountsEndpoint = "/computeMetadata/v1/instance/service-accounts/"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if serviceAccountsEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			bytes, err := json.Marshal([]string{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			fmt.Fprintln(w, string(bytes))
		} else {
			http.Error(w, "", http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	provider := &ContainerRegistryProvider{
		MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
	}

	if provider.Enabled() {
		t.Errorf("Provider is unexpectedly enabled")
	}
}

func ContainerRegistryNoStorageScope(t *testing.T) {
	t.Parallel()
	const (
		serviceAccountsEndpoint = "/computeMetadata/v1/instance/service-accounts/"
		defaultEndpoint         = "/computeMetadata/v1/instance/service-accounts/default/"
		scopeEndpoint           = defaultEndpoint + "scopes"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if scopeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `["https://www.googleapis.com/auth/compute.read_write"]`)
		} else if serviceAccountsEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, "default/\ncustom")
		} else {
			http.Error(w, "", http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	provider := &ContainerRegistryProvider{
		MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
	}

	if provider.Enabled() {
		t.Errorf("Provider is unexpectedly enabled")
	}
}

func ComputePlatformScopeSubstitutesStorageScope(t *testing.T) {
	t.Parallel()
	const (
		serviceAccountsEndpoint = "/computeMetadata/v1/instance/service-accounts/"
		defaultEndpoint         = "/computeMetadata/v1/instance/service-accounts/default/"
		scopeEndpoint           = defaultEndpoint + "scopes"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only serve the URL key and the value endpoint
		if scopeEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `["https://www.googleapis.com/auth/compute.read_write","https://www.googleapis.com/auth/cloud-platform.read-only"]`)
		} else if serviceAccountsEndpoint == r.URL.Path {
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, "default/\ncustom")
		} else {
			http.Error(w, "", http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	provider := &ContainerRegistryProvider{
		MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
	}

	if !provider.Enabled() {
		t.Errorf("Provider is unexpectedly disabled")
	}
}

func AllProvidersNoMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "", http.StatusNotFound)
	}))
	defer server.Close()

	// Make a transport that reroutes all traffic to the example server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(server.URL + req.URL.Path)
		},
	})

	providers := []credentialprovider.DockerConfigProvider{
		&DockerConfigKeyProvider{
			MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
		},
		&DockerConfigURLKeyProvider{
			MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
		},
		&ContainerRegistryProvider{
			MetadataProvider: MetadataProvider{Client: &http.Client{Transport: transport}},
		},
	}

	for _, provider := range providers {
		if provider.Enabled() {
			t.Errorf("Provider %s is unexpectedly enabled", reflect.TypeOf(provider).String())
		}
	}
}
