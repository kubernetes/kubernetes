/*
Copyright 2023 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime/serializer"
	api "k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/install"
	externalapi "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme, serializer.EnableStrict)
)

func init() {
	install.Install(scheme)
}

func LoadFromFile(file string) (*api.AuthorizationConfiguration, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return LoadFromData(data)
}

func LoadFromReader(reader io.Reader) (*api.AuthorizationConfiguration, error) {
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

func LoadFromData(data []byte) (*api.AuthorizationConfiguration, error) {
	if len(data) == 0 {
		// no config provided, return default
		externalConfig := &externalapi.AuthorizationConfiguration{}
		scheme.Default(externalConfig)
		internalConfig := &api.AuthorizationConfiguration{}
		if err := scheme.Convert(externalConfig, internalConfig, nil); err != nil {
			return nil, err
		}
		return internalConfig, nil
	}

	decodedObj, err := runtime.Decode(codecs.UniversalDecoder(), data)
	if err != nil {
		return nil, err
	}
	configuration, ok := decodedObj.(*api.AuthorizationConfiguration)
	if !ok {
		return nil, fmt.Errorf("expected AuthorizationConfiguration, got %T", decodedObj)
	}
	return configuration, nil
}
