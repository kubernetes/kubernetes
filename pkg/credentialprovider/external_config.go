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

package credentialprovider

import (
	"fmt"
	"io/ioutil"

	"k8s.io/kubernetes/pkg/credentialprovider/apis/registrycredentials"
	"k8s.io/kubernetes/pkg/credentialprovider/apis/registrycredentials/v1alpha1"
)

func readExternalProviderConfig(configPath string) (*registrycredentials.RegistryCredentialConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("RegistryCredentialConfigPath is empty when trying to read external registry credential provider config.")
	}

	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("unable to read external registry credential provider configuration from %q [%v]", configPath, err)
	}

	return decode(data)
}

func decode(serialized []byte) (*registrycredentials.RegistryCredentialConfig, error) {
	obj, gvk, err := codecs.UniversalDecoder(v1alpha1.SchemeGroupVersion).Decode(serialized, nil, nil)
	if err != nil {
		return nil, err
	}

	if gvk.Kind != "RegistryCredentialConfig" {
		return nil, fmt.Errorf("failed to decode %q (missing Kind)", gvk.Kind)
	}
	config, err := scheme.ConvertToVersion(obj, registrycredentials.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}
	if internalConfig, ok := config.(*registrycredentials.RegistryCredentialConfig); ok {
		return internalConfig, nil
	}
	return nil, fmt.Errorf("unable to convert %T to *RegistryCredentialConfig", config)
}
