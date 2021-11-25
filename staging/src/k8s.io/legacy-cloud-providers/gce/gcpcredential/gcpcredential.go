/*
Copyright 2020 The Kubernetes Authors.

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

package gcpcredential

import (
	"encoding/json"
	"net/http"
	"strings"

	"k8s.io/cloud-provider/credentialconfig"
	"k8s.io/klog/v2"
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
	metadataToken      = metadataURL + "instance/service-accounts/default/token"
	metadataEmail      = metadataURL + "instance/service-accounts/default/email"
	// StorageScopePrefix is the prefix checked by ContainerRegistryProvider.Enabled.
	StorageScopePrefix       = "https://www.googleapis.com/auth/devstorage"
	cloudPlatformScopePrefix = "https://www.googleapis.com/auth/cloud-platform"
	defaultServiceAccount    = "default/"
)

// GCEProductNameFile is the product file path that contains the cloud service name.
// This is a variable instead of a const to enable testing.
var GCEProductNameFile = "/sys/class/dmi/id/product_name"

// For these urls, the parts of the host name can be glob, for example '*.gcr.io" will match
// "foo.gcr.io" and "bar.gcr.io".
var containerRegistryUrls = []string{"container.cloud.google.com", "gcr.io", "*.gcr.io", "*.pkg.dev"}

var metadataHeader = &http.Header{
	"Metadata-Flavor": []string{"Google"},
}

// ProvideConfigKey implements a dockercfg-based authentication flow.
func ProvideConfigKey(client *http.Client, image string) credentialconfig.RegistryConfig {
	// Read the contents of the google-dockercfg metadata key and
	// parse them as an alternate .dockercfg
	if cfg, err := ReadDockerConfigFileFromURL(DockerConfigKey, client, metadataHeader); err != nil {
		klog.Errorf("while reading 'google-dockercfg' metadata: %v", err)
	} else {
		return cfg
	}

	return credentialconfig.RegistryConfig{}
}

// ProvideURLKey implements a dockercfg-url-based authentication flow.
func ProvideURLKey(client *http.Client, image string) credentialconfig.RegistryConfig {
	// Read the contents of the google-dockercfg-url key and load a .dockercfg from there
	if url, err := ReadURL(DockerConfigURLKey, client, metadataHeader); err != nil {
		klog.Errorf("while reading 'google-dockercfg-url' metadata: %v", err)
	} else {
		if strings.HasPrefix(string(url), "http") {
			if cfg, err := ReadDockerConfigFileFromURL(string(url), client, nil); err != nil {
				klog.Errorf("while reading 'google-dockercfg-url'-specified url: %s, %v", string(url), err)
			} else {
				return cfg
			}
		} else {
			// TODO(mattmoor): support reading alternate scheme URLs (e.g. gs:// or s3://)
			klog.Errorf("Unsupported URL scheme: %s", string(url))
		}
	}

	return credentialconfig.RegistryConfig{}
}

// TokenBlob is used to decode the JSON blob containing an access token
// that is returned by GCE metadata.
type TokenBlob struct {
	AccessToken string `json:"access_token"`
}

// ProvideContainerRegistry implements a gcr.io-based authentication flow.
func ProvideContainerRegistry(client *http.Client, image string) credentialconfig.RegistryConfig {
	cfg := credentialconfig.RegistryConfig{}

	tokenJSONBlob, err := ReadURL(metadataToken, client, metadataHeader)
	if err != nil {
		klog.Errorf("while reading access token endpoint: %v", err)
		return cfg
	}

	email, err := ReadURL(metadataEmail, client, metadataHeader)
	if err != nil {
		klog.Errorf("while reading email endpoint: %v", err)
		return cfg
	}

	var parsedBlob TokenBlob
	if err := json.Unmarshal([]byte(tokenJSONBlob), &parsedBlob); err != nil {
		klog.Errorf("error while parsing json blob of length %d", len(tokenJSONBlob))
		return cfg
	}

	entry := credentialconfig.RegistryConfigEntry{
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
