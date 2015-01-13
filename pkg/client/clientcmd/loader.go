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

	"github.com/ghodss/yaml"
	"github.com/imdario/mergo"

	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	clientcmdlatest "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api/latest"
)

const (
	RecommendedConfigPathFlag   = "kubeconfig"
	RecommendedConfigPathEnvVar = "KUBECONFIG"
)

// ClientConfigLoadingRules is a struct that calls our specific locations that are used for merging together a Config
type ClientConfigLoadingRules struct {
	CommandLinePath      string
	EnvVarPath           string
	CurrentDirectoryPath string
	HomeDirectoryPath    string
}

// NewClientConfigLoadingRules returns a ClientConfigLoadingRules object with default fields filled in.  You are not required to
// use this constructor
func NewClientConfigLoadingRules() *ClientConfigLoadingRules {
	return &ClientConfigLoadingRules{
		CurrentDirectoryPath: ".kubeconfig",
		HomeDirectoryPath:    os.Getenv("HOME") + "/.kube/.kubeconfig",
	}
}

// Load takes the loading rules and merges together a Config object based on following order.
//   1.  CommandLinePath
//   2.  EnvVarPath
//   3.  CurrentDirectoryPath
//   4.  HomeDirectoryPath
// Empty filenames are ignored.  Files with non-deserializable content produced errors.
// The first file to set a particular value or map key wins and the value or map key is never changed.
// This means that the first file to set CurrentContext will have its context preserved.  It also means
// that if two files specify a "red-user", only values from the first file's red-user are used.  Even
// non-conflicting entries from the second file's "red-user" are discarded.
func (rules *ClientConfigLoadingRules) Load() (*clientcmdapi.Config, error) {
	config := clientcmdapi.NewConfig()

	mergeConfigWithFile(config, rules.CommandLinePath)
	mergeConfigWithFile(config, rules.EnvVarPath)
	mergeConfigWithFile(config, rules.CurrentDirectoryPath)
	mergeConfigWithFile(config, rules.HomeDirectoryPath)

	return config, nil
}

func mergeConfigWithFile(startingConfig *clientcmdapi.Config, filename string) error {
	if len(filename) == 0 {
		// no work to do
		return nil
	}

	config, err := LoadFromFile(filename)
	if err != nil {
		return err
	}

	mergo.Merge(startingConfig, config)

	return nil
}

// LoadFromFile takes a filename and deserializes the contents into Config object
func LoadFromFile(filename string) (*clientcmdapi.Config, error) {
	config := &clientcmdapi.Config{}

	kubeconfigBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
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
