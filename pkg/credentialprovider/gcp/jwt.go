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
	"io/ioutil"
	"time"

	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/jwt"
	"k8s.io/kubernetes/pkg/credentialprovider"

	"github.com/spf13/pflag"
)

const (
	storageReadOnlyScope = "https://www.googleapis.com/auth/devstorage.read_only"
)

var (
	flagJwtFile = pflag.String("google-json-key", "",
		"The Google Cloud Platform Service Account JSON Key to use for authentication.")
)

// A DockerConfigProvider that reads its configuration from Google
// Compute Engine metadata.
type jwtProvider struct {
	path     *string
	config   *jwt.Config
	tokenUrl string
}

// init registers the various means by which credentials may
// be resolved on GCP.
func init() {
	credentialprovider.RegisterCredentialProvider("google-jwt-key",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &jwtProvider{
				path: flagJwtFile,
			},
			Lifetime: 30 * time.Minute,
		})
}

// Enabled implements DockerConfigProvider for the JSON Key based implementation.
func (j *jwtProvider) Enabled() bool {
	if *j.path == "" {
		return false
	}

	data, err := ioutil.ReadFile(*j.path)
	if err != nil {
		glog.Errorf("while reading file %s got %v", *j.path, err)
		return false
	}
	config, err := google.JWTConfigFromJSON(data, storageReadOnlyScope)
	if err != nil {
		glog.Errorf("while parsing %s data got %v", *j.path, err)
		return false
	}

	j.config = config
	if j.tokenUrl != "" {
		j.config.TokenURL = j.tokenUrl
	}
	return true
}

// Provide implements DockerConfigProvider
func (j *jwtProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}

	ts := j.config.TokenSource(oauth2.NoContext)
	token, err := ts.Token()
	if err != nil {
		glog.Errorf("while exchanging json key %s for access token %v", *j.path, err)
		return cfg
	}
	if !token.Valid() {
		glog.Errorf("Got back invalid token: %v", token)
		return cfg
	}

	entry := credentialprovider.DockerConfigEntry{
		Username: "_token",
		Password: token.AccessToken,
		Email:    j.config.Email,
	}

	// Add our entry for each of the supported container registry URLs
	for _, k := range containerRegistryUrls {
		cfg[k] = entry
	}
	return cfg
}
