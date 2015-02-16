/*
Copyright 2014 Google Inc. All rights reserved.

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

package clientcmd

import (
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/ghodss/yaml"

	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	clientcmdlatest "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api/latest"
)

const (
	RecommendedConfigPathFlag   = "kubeconfig"
	RecommendedConfigPathEnvVar = "KUBECONFIG"
)

// ClientConfigLoadingOrder is a struct that calls our specific locations that are used for merging together a Config
type ClientConfigLoadingOrder []string

// DefaultClientConfigLoadingOrder returns a ClientConfigLoadingOrder object with default values.  Index 0, is empty to allow callers
// to bind a command line flag if they like.  You are not required to use this constructor
func DefaultClientConfigLoadingOrder() ClientConfigLoadingOrder {
	return ClientConfigLoadingOrder([]string{
		"", // usually bound to "--kubeconfig" flag
		os.Getenv(RecommendedConfigPathEnvVar),
		os.Getenv("HOME") + "/.kube/.kubeconfig",
	})
}

// Load takes the loading order and returns the kubeconfig from the first existing file along with the filename used.
// If no file in the order exists, it returns an empty config.
// Empty filenames are ignored.  Missing files are ignored.  Files with non-deserializable content produced errors.
func (order ClientConfigLoadingOrder) Load() (*clientcmdapi.Config, string, error) {
	for _, file := range order {
		if len(file) > 0 {
			config, err := LoadFromFile(file)
			if err != nil {
				if os.IsNotExist(err) {
					// the config file didn't exist, try the next one
					continue
				}

				return nil, file, err
			}

			if err := resolveLocalPaths(file, config); err != nil {
				return nil, file, err
			}

			return config, file, nil
		}
	}

	return clientcmdapi.NewConfig(), "", nil
}

// resolveLocalPaths resolves all relative paths in the config object with respect to the parent directory of the filename
// this cannot be done directly inside of LoadFromFile because doing so there would make it impossible to load a file without
// modification of its contents.
func resolveLocalPaths(filename string, config *clientcmdapi.Config) error {
	if len(filename) == 0 {
		return nil
	}

	configDir, err := filepath.Abs(filepath.Dir(filename))
	if err != nil {
		return err
	}

	resolvedClusters := make(map[string]clientcmdapi.Cluster)
	for key, cluster := range config.Clusters {
		cluster.CertificateAuthority = resolveLocalPath(configDir, cluster.CertificateAuthority)
		resolvedClusters[key] = cluster
	}
	config.Clusters = resolvedClusters

	resolvedAuthInfos := make(map[string]clientcmdapi.AuthInfo)
	for key, authInfo := range config.AuthInfos {
		authInfo.AuthPath = resolveLocalPath(configDir, authInfo.AuthPath)
		authInfo.ClientCertificate = resolveLocalPath(configDir, authInfo.ClientCertificate)
		authInfo.ClientKey = resolveLocalPath(configDir, authInfo.ClientKey)
		resolvedAuthInfos[key] = authInfo
	}
	config.AuthInfos = resolvedAuthInfos

	return nil
}

// resolveLocalPath makes the path absolute with respect to the startingDir
func resolveLocalPath(startingDir, path string) string {
	if len(path) == 0 {
		return path
	}
	if filepath.IsAbs(path) {
		return path
	}

	return filepath.Join(startingDir, path)
}

// LoadFromFile takes a filename and deserializes the contents into Config object
func LoadFromFile(filename string) (*clientcmdapi.Config, error) {
	config := &clientcmdapi.Config{}

	kubeconfigBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// decode doesn't allow empty files, but they are valid for this sort of an operation.  Empty files can still be loaded,
	// they just have no contents.  This is what would be expected if a user cleared the contents of .kubeconfig files or just
	// used a touch to create one in a particular location
	if len(kubeconfigBytes) == 0 {
		return config, nil
	}

	if err := clientcmdlatest.Codec.DecodeInto(kubeconfigBytes, config); err != nil {
		return nil, err
	}

	return config, nil
}

// WriteToFile serializes the config to yaml and writes it out to a file.  If no present, it creates the file with 0644.  If it is present
// it stomps the contents
func WriteToFile(config clientcmdapi.Config, filename string) error {
	json, err := clientcmdlatest.Codec.Encode(&config)
	if err != nil {
		return err
	}

	content, err := yaml.JSONToYAML(json)
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(filename, content, 0644); err != nil {
		return err
	}

	return nil
}
