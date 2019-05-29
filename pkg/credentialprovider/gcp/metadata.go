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

package gcp_credentials

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

const (
	metadataUrl              = "http://metadata.google.internal./computeMetadata/v1/"
	metadataAttributes       = metadataUrl + "instance/attributes/"
	dockerConfigKey          = metadataAttributes + "google-dockercfg"
	dockerConfigUrlKey       = metadataAttributes + "google-dockercfg-url"
	serviceAccounts          = metadataUrl + "instance/service-accounts/"
	metadataScopes           = metadataUrl + "instance/service-accounts/default/scopes"
	metadataToken            = metadataUrl + "instance/service-accounts/default/token"
	metadataEmail            = metadataUrl + "instance/service-accounts/default/email"
	storageScopePrefix       = "https://www.googleapis.com/auth/devstorage"
	cloudPlatformScopePrefix = "https://www.googleapis.com/auth/cloud-platform"
	defaultServiceAccount    = "default/"
)

// Product file path that contains the cloud service name.
// This is a variable instead of a const to enable testing.
var gceProductNameFile = "/sys/class/dmi/id/product_name"

// For these urls, the parts of the host name can be glob, for example '*.gcr.io" will match
// "foo.gcr.io" and "bar.gcr.io".
var containerRegistryUrls = []string{"container.cloud.google.com", "gcr.io", "*.gcr.io"}

var metadataHeader = &http.Header{
	"Metadata-Flavor": []string{"Google"},
}

// A DockerConfigProvider that reads its configuration from Google
// Compute Engine metadata.
type metadataProvider struct {
	Client *http.Client
}

// A DockerConfigProvider that reads its configuration from a specific
// Google Compute Engine metadata key: 'google-dockercfg'.
type dockerConfigKeyProvider struct {
	metadataProvider
}

// A DockerConfigProvider that reads its configuration from a URL read from
// a specific Google Compute Engine metadata key: 'google-dockercfg-url'.
type dockerConfigUrlKeyProvider struct {
	metadataProvider
}

// A DockerConfigProvider that provides a dockercfg with:
//    Username: "_token"
//    Password: "{access token from metadata}"
type containerRegistryProvider struct {
	metadataProvider
}

// init registers the various means by which credentials may
// be resolved on GCP.
func init() {
	tr := utilnet.SetTransportDefaults(&http.Transport{})
	metadataHTTPClientTimeout := time.Second * 10
	httpClient := &http.Client{
		Transport: tr,
		Timeout:   metadataHTTPClientTimeout,
	}
	credentialprovider.RegisterCredentialProvider("google-dockercfg",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &dockerConfigKeyProvider{
				metadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-dockercfg-url",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &dockerConfigUrlKeyProvider{
				metadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-container-registry",
		// Never cache this.  The access token is already
		// cached by the metadata service.
		&containerRegistryProvider{
			metadataProvider{Client: httpClient},
		})
}

// Returns true if it finds a local GCE VM.
// Looks at a product file that is an undocumented API.
func onGCEVM() bool {
	data, err := ioutil.ReadFile(gceProductNameFile)
	if err != nil {
		klog.V(2).Infof("Error while reading product_name: %v", err)
		return false
	}
	name := strings.TrimSpace(string(data))
	return name == "Google" || name == "Google Compute Engine"
}

// Enabled implements DockerConfigProvider for all of the Google implementations.
func (g *metadataProvider) Enabled() bool {
	return onGCEVM()
}

// LazyProvide implements DockerConfigProvider. Should never be called.
func (g *dockerConfigKeyProvider) LazyProvide(image string) *credentialprovider.DockerConfigEntry {
	return nil
}

// Provide implements DockerConfigProvider
func (g *dockerConfigKeyProvider) Provide(image string) credentialprovider.DockerConfig {
	// Read the contents of the google-dockercfg metadata key and
	// parse them as an alternate .dockercfg
	if cfg, err := credentialprovider.ReadDockerConfigFileFromUrl(dockerConfigKey, g.Client, metadataHeader); err != nil {
		klog.Errorf("while reading 'google-dockercfg' metadata: %v", err)
	} else {
		return cfg
	}

	return credentialprovider.DockerConfig{}
}

// LazyProvide implements DockerConfigProvider. Should never be called.
func (g *dockerConfigUrlKeyProvider) LazyProvide(image string) *credentialprovider.DockerConfigEntry {
	return nil
}

// Provide implements DockerConfigProvider
func (g *dockerConfigUrlKeyProvider) Provide(image string) credentialprovider.DockerConfig {
	// Read the contents of the google-dockercfg-url key and load a .dockercfg from there
	if url, err := credentialprovider.ReadUrl(dockerConfigUrlKey, g.Client, metadataHeader); err != nil {
		klog.Errorf("while reading 'google-dockercfg-url' metadata: %v", err)
	} else {
		if strings.HasPrefix(string(url), "http") {
			if cfg, err := credentialprovider.ReadDockerConfigFileFromUrl(string(url), g.Client, nil); err != nil {
				klog.Errorf("while reading 'google-dockercfg-url'-specified url: %s, %v", string(url), err)
			} else {
				return cfg
			}
		} else {
			// TODO(mattmoor): support reading alternate scheme URLs (e.g. gs:// or s3://)
			klog.Errorf("Unsupported URL scheme: %s", string(url))
		}
	}

	return credentialprovider.DockerConfig{}
}

// runWithBackoff runs input function `f` with an exponential backoff.
// Note that this method can block indefinitely.
func runWithBackoff(f func() ([]byte, error)) []byte {
	var backoff = 100 * time.Millisecond
	const maxBackoff = time.Minute
	for {
		value, err := f()
		if err == nil {
			return value
		}
		time.Sleep(backoff)
		backoff = backoff * 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// Enabled implements a special metadata-based check, which verifies the
// storage scope is available on the GCE VM.
// If running on a GCE VM, check if 'default' service account exists.
// If it does not exist, assume that registry is not enabled.
// If default service account exists, check if relevant scopes exist in the default service account.
// The metadata service can become temporarily inaccesible. Hence all requests to the metadata
// service will be retried until the metadata server returns a `200`.
// It is expected that "http://metadata.google.internal./computeMetadata/v1/instance/service-accounts/" will return a `200`
// and "http://metadata.google.internal./computeMetadata/v1/instance/service-accounts/default/scopes" will also return `200`.
// More information on metadata service can be found here - https://cloud.google.com/compute/docs/storing-retrieving-metadata
func (g *containerRegistryProvider) Enabled() bool {
	if !onGCEVM() {
		return false
	}
	// Given that we are on GCE, we should keep retrying until the metadata server responds.
	value := runWithBackoff(func() ([]byte, error) {
		value, err := credentialprovider.ReadUrl(serviceAccounts, g.Client, metadataHeader)
		if err != nil {
			klog.V(2).Infof("Failed to Get service accounts from gce metadata server: %v", err)
		}
		return value, err
	})
	// We expect the service account to return a list of account directories separated by newlines, e.g.,
	//   sv-account-name1/
	//   sv-account-name2/
	// ref: https://cloud.google.com/compute/docs/storing-retrieving-metadata
	defaultServiceAccountExists := false
	for _, sa := range strings.Split(string(value), "\n") {
		if strings.TrimSpace(sa) == defaultServiceAccount {
			defaultServiceAccountExists = true
			break
		}
	}
	if !defaultServiceAccountExists {
		klog.V(2).Infof("'default' service account does not exist. Found following service accounts: %q", string(value))
		return false
	}
	url := metadataScopes + "?alt=json"
	value = runWithBackoff(func() ([]byte, error) {
		value, err := credentialprovider.ReadUrl(url, g.Client, metadataHeader)
		if err != nil {
			klog.V(2).Infof("Failed to Get scopes in default service account from gce metadata server: %v", err)
		}
		return value, err
	})
	var scopes []string
	if err := json.Unmarshal(value, &scopes); err != nil {
		klog.Errorf("Failed to unmarshal scopes: %v", err)
		return false
	}
	for _, v := range scopes {
		// cloudPlatformScope implies storage scope.
		if strings.HasPrefix(v, storageScopePrefix) || strings.HasPrefix(v, cloudPlatformScopePrefix) {
			return true
		}
	}
	klog.Warningf("Google container registry is disabled, no storage scope is available: %s", value)
	return false
}

// tokenBlob is used to decode the JSON blob containing an access token
// that is returned by GCE metadata.
type tokenBlob struct {
	AccessToken string `json:"access_token"`
}

// LazyProvide implements DockerConfigProvider. Should never be called.
func (g *containerRegistryProvider) LazyProvide(image string) *credentialprovider.DockerConfigEntry {
	return nil
}

// Provide implements DockerConfigProvider
func (g *containerRegistryProvider) Provide(image string) credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}

	tokenJsonBlob, err := credentialprovider.ReadUrl(metadataToken, g.Client, metadataHeader)
	if err != nil {
		klog.Errorf("while reading access token endpoint: %v", err)
		return cfg
	}

	email, err := credentialprovider.ReadUrl(metadataEmail, g.Client, metadataHeader)
	if err != nil {
		klog.Errorf("while reading email endpoint: %v", err)
		return cfg
	}

	var parsedBlob tokenBlob
	if err := json.Unmarshal([]byte(tokenJsonBlob), &parsedBlob); err != nil {
		klog.Errorf("while parsing json blob %s: %v", tokenJsonBlob, err)
		return cfg
	}

	entry := credentialprovider.DockerConfigEntry{
		Username: "_token",
		Password: parsedBlob.AccessToken,
		Email:    string(email),
	}

	// Add our entry for each of the supported container registry URLs
	for _, k := range containerRegistryUrls {
		cfg[k] = entry
	}
	return cfg
}
