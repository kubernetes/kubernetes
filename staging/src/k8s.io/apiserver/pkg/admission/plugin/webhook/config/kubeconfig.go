/*
Copyright 2017 The Kubernetes Authors.

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

package config

import (
	"io"

	"k8s.io/apimachinery/pkg/util/yaml"
)

// AdmissionConfig holds config data that is unique to each API server.
type AdmissionConfig struct {
	// KubeConfigFile is the path to the kubeconfig file.
	KubeConfigFile string `json:"kubeConfigFile"`
}

// LoadConfig extract the KubeConfigFile from configFile
func LoadConfig(configFile io.Reader) (string, error) {
	var kubeconfigFile string
	if configFile != nil {
		// TODO: move this to a versioned configuration file format
		var config AdmissionConfig
		d := yaml.NewYAMLOrJSONDecoder(configFile, 4096)
		err := d.Decode(&config)
		if err != nil {
			return "", err
		}
		kubeconfigFile = config.KubeConfigFile
	}
	return kubeconfigFile, nil
}
