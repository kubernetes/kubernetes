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
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/cloud-provider/credentialconfig"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/legacy-cloud-providers/gce/gcpcredential"
)

const (
	metadataURL        = "http://metadata.google.internal./computeMetadata/v1/"
	metadataAttributes = metadataURL + "instance/attributes/"
	// DockerConfigKey is the URL of the dockercfg metadata key used by DockerConfigKeyProvider.
	DockerConfigKey = metadataAttributes + "google-dockercfg"
	// DockerConfigURLKey is the URL of the dockercfg metadata key used by DockerConfigURLKeyProvider.
	DockerConfigURLKey = metadataAttributes + "google-dockercfg-url"
	serviceAccounts    = metadataURL + "instance/service-accounts/"
	metadataScopes     = metadataURL + "instance/service-accounts/default/scopes"
	// StorageScopePrefix is the prefix checked by ContainerRegistryProvider.Enabled.
	StorageScopePrefix       = "https://www.googleapis.com/auth/devstorage"
	cloudPlatformScopePrefix = "https://www.googleapis.com/auth/cloud-platform"
	defaultServiceAccount    = "default/"
)

// gceProductNameFile is the product file path that contains the cloud service name.
// This is a variable instead of a const to enable testing.
var gceProductNameFile = "/sys/class/dmi/id/product_name"

var metadataHeader = &http.Header{
	"Metadata-Flavor": []string{"Google"},
}

var warnOnce sync.Once

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
			Provider: &DockerConfigKeyProvider{
				MetadataProvider: MetadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-dockercfg-url",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &DockerConfigURLKeyProvider{
				MetadataProvider: MetadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-container-registry",
		// Never cache this.  The access token is already
		// cached by the metadata service.
		&ContainerRegistryProvider{
			MetadataProvider: MetadataProvider{Client: httpClient},
		})
}

// MetadataProvider is a DockerConfigProvider that reads its configuration from Google
// Compute Engine metadata.
type MetadataProvider struct {
	Client *http.Client
}

// DockerConfigKeyProvider is a DockerConfigProvider that reads its configuration from a specific
// Google Compute Engine metadata key: 'google-dockercfg'.
type DockerConfigKeyProvider struct {
	MetadataProvider
}

// DockerConfigURLKeyProvider is a DockerConfigProvider that reads its configuration from a URL read from
// a specific Google Compute Engine metadata key: 'google-dockercfg-url'.
type DockerConfigURLKeyProvider struct {
	MetadataProvider
}

// ContainerRegistryProvider is a DockerConfigProvider that provides a dockercfg with:
//
//	Username: "_token"
//	Password: "{access token from metadata}"
type ContainerRegistryProvider struct {
	MetadataProvider
}

// Returns true if it finds a local GCE VM.
// Looks at a product file that is an undocumented API.
func onGCEVM() bool {
	var name string

	if runtime.GOOS == "windows" {
		data, err := exec.Command("wmic", "computersystem", "get", "model").Output()
		if err != nil {
			return false
		}
		fields := strings.Split(strings.TrimSpace(string(data)), "\r\n")
		if len(fields) != 2 {
			klog.V(2).Infof("Received unexpected value retrieving system model: %q", string(data))
			return false
		}
		name = fields[1]
	} else {
		data, err := os.ReadFile(gceProductNameFile)
		if err != nil {
			klog.V(2).Infof("Error while reading product_name: %v", err)
			return false
		}
		name = strings.TrimSpace(string(data))
	}
	return name == "Google" || name == "Google Compute Engine"
}

// Enabled implements DockerConfigProvider for all of the Google implementations.
func (g *MetadataProvider) Enabled() bool {
	onGCE := onGCEVM()
	if !onGCE {
		return false
	}
	if credentialprovider.AreLegacyCloudCredentialProvidersDisabled() {
		warnOnce.Do(func() {
			klog.V(4).Infof("GCP credential provider is now disabled. Please refer to sig-cloud-provider for guidance on external credential provider integration for GCP")
		})
		return false
	}
	return true
}

// Provide implements DockerConfigProvider
func (g *DockerConfigKeyProvider) Provide(image string) credentialprovider.DockerConfig {
	return registryToDocker(gcpcredential.ProvideConfigKey(g.Client, image))
}

// Provide implements DockerConfigProvider
func (g *DockerConfigURLKeyProvider) Provide(image string) credentialprovider.DockerConfig {
	return registryToDocker(gcpcredential.ProvideURLKey(g.Client, image))
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
func (g *ContainerRegistryProvider) Enabled() bool {
	if !onGCEVM() {
		return false
	}

	if credentialprovider.AreLegacyCloudCredentialProvidersDisabled() {
		warnOnce.Do(func() {
			klog.V(4).Infof("GCP credential provider is now disabled. Please refer to sig-cloud-provider for guidance on external credential provider integration for GCP")
		})
		return false
	}

	// Given that we are on GCE, we should keep retrying until the metadata server responds.
	value := runWithBackoff(func() ([]byte, error) {
		value, err := gcpcredential.ReadURL(serviceAccounts, g.Client, metadataHeader)
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
		value, err := gcpcredential.ReadURL(url, g.Client, metadataHeader)
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
		if strings.HasPrefix(v, StorageScopePrefix) || strings.HasPrefix(v, cloudPlatformScopePrefix) {
			return true
		}
	}
	klog.Warningf("Google container registry is disabled, no storage scope is available: %s", value)
	return false
}

// Provide implements DockerConfigProvider
func (g *ContainerRegistryProvider) Provide(image string) credentialprovider.DockerConfig {
	return registryToDocker(gcpcredential.ProvideContainerRegistry(g.Client, image))
}

func registryToDocker(registryConfig credentialconfig.RegistryConfig) credentialprovider.DockerConfig {
	dockerConfig := credentialprovider.DockerConfig{}
	for k, v := range registryConfig {
		dockerConfig[k] = credentialprovider.DockerConfigEntry{
			Username: v.Username,
			Password: v.Password,
			Email:    v.Email,
		}
	}
	return dockerConfig
}
