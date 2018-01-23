/*
Copyright 2015 The Kubernetes Authors.

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

package api

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
)

func init() {
	sDec, _ := base64.StdEncoding.DecodeString("REDACTED+")
	redactedBytes = []byte(string(sDec))
}

// IsConfigEmpty returns true if the config is empty.
func IsConfigEmpty(config *Config) bool {
	return len(config.AuthInfos) == 0 && len(config.Clusters) == 0 && len(config.Contexts) == 0 &&
		len(config.CurrentContext) == 0 &&
		len(config.Preferences.Extensions) == 0 && !config.Preferences.Colors &&
		len(config.Extensions) == 0
}

// MinifyConfig read the current context and uses that to keep only the relevant pieces of config
// This is useful for making secrets based on kubeconfig files
func MinifyConfig(config *Config) error {
	if len(config.CurrentContext) == 0 {
		return errors.New("current-context must exist in order to minify")
	}

	currContext, exists := config.Contexts[config.CurrentContext]
	if !exists {
		return fmt.Errorf("cannot locate context %v", config.CurrentContext)
	}

	newContexts := map[string]*Context{}
	newContexts[config.CurrentContext] = currContext

	newClusters := map[string]*Cluster{}
	if len(currContext.Cluster) > 0 {
		if _, exists := config.Clusters[currContext.Cluster]; !exists {
			return fmt.Errorf("cannot locate cluster %v", currContext.Cluster)
		}

		newClusters[currContext.Cluster] = config.Clusters[currContext.Cluster]
	}

	newAuthInfos := map[string]*AuthInfo{}
	if len(currContext.AuthInfo) > 0 {
		if _, exists := config.AuthInfos[currContext.AuthInfo]; !exists {
			return fmt.Errorf("cannot locate user %v", currContext.AuthInfo)
		}

		newAuthInfos[currContext.AuthInfo] = config.AuthInfos[currContext.AuthInfo]
	}

	config.AuthInfos = newAuthInfos
	config.Clusters = newClusters
	config.Contexts = newContexts

	return nil
}

var redactedBytes []byte

// Flatten redacts raw data entries from the config object for a human-readable view.
func ShortenConfig(config *Config) {
	// trick json encoder into printing a human readable string in the raw data
	// by base64 decoding what we want to print. Relies on implementation of
	// http://golang.org/pkg/encoding/json/#Marshal using base64 to encode []byte
	for key, authInfo := range config.AuthInfos {
		if len(authInfo.ClientKeyData) > 0 {
			authInfo.ClientKeyData = redactedBytes
		}
		if len(authInfo.ClientCertificateData) > 0 {
			authInfo.ClientCertificateData = redactedBytes
		}
		config.AuthInfos[key] = authInfo
	}
	for key, cluster := range config.Clusters {
		if len(cluster.CertificateAuthorityData) > 0 {
			cluster.CertificateAuthorityData = redactedBytes
		}
		config.Clusters[key] = cluster
	}
}

// Flatten changes the config object into a self contained config (useful for making secrets)
func FlattenConfig(config *Config) error {
	for key, authInfo := range config.AuthInfos {
		baseDir, err := MakeAbs(path.Dir(authInfo.LocationOfOrigin), "")
		if err != nil {
			return err
		}

		if err := FlattenContent(&authInfo.ClientCertificate, &authInfo.ClientCertificateData, baseDir); err != nil {
			return err
		}
		if err := FlattenContent(&authInfo.ClientKey, &authInfo.ClientKeyData, baseDir); err != nil {
			return err
		}

		config.AuthInfos[key] = authInfo
	}
	for key, cluster := range config.Clusters {
		baseDir, err := MakeAbs(path.Dir(cluster.LocationOfOrigin), "")
		if err != nil {
			return err
		}

		if err := FlattenContent(&cluster.CertificateAuthority, &cluster.CertificateAuthorityData, baseDir); err != nil {
			return err
		}

		config.Clusters[key] = cluster
	}

	return nil
}

func FlattenContent(path *string, contents *[]byte, baseDir string) error {
	if len(*path) != 0 {
		if len(*contents) > 0 {
			return errors.New("cannot have values for both path and contents")
		}

		var err error
		absPath := ResolvePath(*path, baseDir)
		*contents, err = ioutil.ReadFile(absPath)
		if err != nil {
			return err
		}

		*path = ""
	}

	return nil
}

// ResolvePath returns the path as an absolute paths, relative to the given base directory
func ResolvePath(path string, base string) string {
	// Don't resolve empty paths
	if len(path) > 0 {
		// Don't resolve absolute paths
		if !filepath.IsAbs(path) {
			return filepath.Join(base, path)
		}
	}

	return path
}

func MakeAbs(path, base string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}
	if len(base) == 0 {
		cwd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		base = cwd
	}
	return filepath.Join(base, path), nil
}
