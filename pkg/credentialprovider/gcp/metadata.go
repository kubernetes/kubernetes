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
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/golang/glog"
)

const (
	metadataUrl        = "http://metadata.google.internal./computeMetadata/v1/"
	metadataAttributes = metadataUrl + "instance/attributes/"
	dockerConfigKey    = metadataAttributes + "google-dockercfg"
	dockerConfigUrlKey = metadataAttributes + "google-dockercfg-url"
	metadataScopes     = metadataUrl + "instance/service-accounts/default/scopes"
	metadataToken      = metadataUrl + "instance/service-accounts/default/token"
	metadataEmail      = metadataUrl + "instance/service-accounts/default/email"
	storageScopePrefix = "https://www.googleapis.com/auth/devstorage"
)

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
	credentialprovider.RegisterCredentialProvider("google-dockercfg",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &dockerConfigKeyProvider{
				metadataProvider{Client: http.DefaultClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-dockercfg-url",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &dockerConfigUrlKeyProvider{
				metadataProvider{Client: http.DefaultClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-container-registry",
		// Never cache this.  The access token is already
		// cached by the metadata service.
		&containerRegistryProvider{
			metadataProvider{Client: http.DefaultClient},
		})
}

// Enabled implements DockerConfigProvider for all of the Google implementations.
func (g *metadataProvider) Enabled() bool {
	_, err := credentialprovider.ReadUrl(metadataUrl, g.Client, metadataHeader)
	return err == nil
}

// Provide implements DockerConfigProvider
func (g *dockerConfigKeyProvider) Provide() credentialprovider.DockerConfig {
	// Read the contents of the google-dockercfg metadata key and
	// parse them as an alternate .dockercfg
	if cfg, err := credentialprovider.ReadDockerConfigFileFromUrl(dockerConfigKey, g.Client, metadataHeader); err != nil {
		glog.Errorf("while reading 'google-dockercfg' metadata: %v", err)
	} else {
		return cfg
	}

	return credentialprovider.DockerConfig{}
}

// Provide implements DockerConfigProvider
func (g *dockerConfigUrlKeyProvider) Provide() credentialprovider.DockerConfig {
	// Read the contents of the google-dockercfg-url key and load a .dockercfg from there
	if url, err := credentialprovider.ReadUrl(dockerConfigUrlKey, g.Client, metadataHeader); err != nil {
		glog.Errorf("while reading 'google-dockercfg-url' metadata: %v", err)
	} else {
		if strings.HasPrefix(string(url), "http") {
			if cfg, err := credentialprovider.ReadDockerConfigFileFromUrl(string(url), g.Client, nil); err != nil {
				glog.Errorf("while reading 'google-dockercfg-url'-specified url: %s, %v", string(url), err)
			} else {
				return cfg
			}
		} else {
			// TODO(mattmoor): support reading alternate scheme URLs (e.g. gs:// or s3://)
			glog.Errorf("Unsupported URL scheme: %s", string(url))
		}
	}

	return credentialprovider.DockerConfig{}
}

// Enabled implements a special metadata-based check, which verifies the
// storage scope is available on the GCE VM.
func (g *containerRegistryProvider) Enabled() bool {
	value, err := credentialprovider.ReadUrl(metadataScopes+"?alt=json", g.Client, metadataHeader)
	if err != nil {
		return false
	}
	var scopes []string
	if err := json.Unmarshal([]byte(value), &scopes); err != nil {
		return false
	}

	for _, v := range scopes {
		if strings.HasPrefix(v, storageScopePrefix) {
			return true
		}
	}
	glog.Warningf("Google container registry is disabled, no storage scope is available: %s", value)
	return false
}

// tokenBlob is used to decode the JSON blob containing an access token
// that is returned by GCE metadata.
type tokenBlob struct {
	AccessToken string `json:"access_token"`
}

// Provide implements DockerConfigProvider
func (g *containerRegistryProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}

	tokenJsonBlob, err := credentialprovider.ReadUrl(metadataToken, g.Client, metadataHeader)
	if err != nil {
		glog.Errorf("while reading access token endpoint: %v", err)
		return cfg
	}

	email, err := credentialprovider.ReadUrl(metadataEmail, g.Client, metadataHeader)
	if err != nil {
		glog.Errorf("while reading email endpoint: %v", err)
		return cfg
	}

	var parsedBlob tokenBlob
	if err := json.Unmarshal([]byte(tokenJsonBlob), &parsedBlob); err != nil {
		glog.Errorf("while parsing json blob %s: %v", tokenJsonBlob, err)
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
