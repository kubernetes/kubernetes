/*
Copyright 2021 The Kubernetes Authors.

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

package load

import (
	"fmt"
	"io"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/pod-security-admission/admission/api"
	"k8s.io/pod-security-admission/admission/api/scheme"
	apiv1beta1 "k8s.io/pod-security-admission/admission/api/v1beta1"
)

func LoadFromFile(file string) (*api.PodSecurityConfiguration, error) {
	if len(file) == 0 {
		// no file specified, use default config
		return LoadFromData(nil)
	}

	// read from file
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return LoadFromData(data)
}

func LoadFromReader(reader io.Reader) (*api.PodSecurityConfiguration, error) {
	if reader == nil {
		// no reader specified, use default config
		return LoadFromData(nil)
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	return LoadFromData(data)
}

func LoadFromData(data []byte) (*api.PodSecurityConfiguration, error) {
	if len(data) == 0 {
		// no config provided, return default
		externalConfig := &apiv1beta1.PodSecurityConfiguration{}
		scheme.Scheme.Default(externalConfig)
		internalConfig := &api.PodSecurityConfiguration{}
		if err := scheme.Scheme.Convert(externalConfig, internalConfig, nil); err != nil {
			return nil, err
		}
		return internalConfig, nil
	}

	decodedObj, err := runtime.Decode(scheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		return nil, err
	}
	configuration, ok := decodedObj.(*api.PodSecurityConfiguration)
	if !ok {
		return nil, fmt.Errorf("expected PodSecurityConfiguration, got %T", decodedObj)
	}
	return configuration, nil
}
