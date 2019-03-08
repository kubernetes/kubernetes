/*
Copyright 2019 The Kubernetes Authors.

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

package nodetaint

import (
	"io"

	"k8s.io/apimachinery/pkg/util/yaml"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// AdmissionConfig holds config data for admission controllers
type AdmissionConfig struct {
	Taints []api.Taint
}

// loadConfiguration loads the provided configuration.
func loadConfiguration(reader io.Reader) (*AdmissionConfig, error) {
	// if no config is provided, return a default configuration
	var admissionConfig AdmissionConfig
	if reader == nil {
		return &admissionConfig, nil
	}
	// we have a config so parse it.
	d := yaml.NewYAMLOrJSONDecoder(reader, 4096)
	if err := d.Decode(&admissionConfig); err != nil {
		return nil, err
	}
	return &admissionConfig, nil
}
