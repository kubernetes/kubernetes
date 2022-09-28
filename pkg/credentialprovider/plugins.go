/*
Copyright 2014 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"sync"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

// All registered credential providers.
var providersMutex sync.Mutex
var providers = make(map[string]DockerConfigProvider)

// RegisterCredentialProvider is called by provider implementations on
// initialization to register themselves, like so:
//
//	func init() {
//	 	RegisterCredentialProvider("name", &myProvider{...})
//	}
func RegisterCredentialProvider(name string, provider DockerConfigProvider) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	_, found := providers[name]
	if found {
		klog.Fatalf("Credential provider %q was registered twice", name)
	}
	klog.V(4).Infof("Registered credential provider %q", name)
	providers[name] = provider
}

// AreLegacyCloudCredentialProvidersDisabled checks if the legacy in-tree cloud
// credential providers have been disabled.
func AreLegacyCloudCredentialProvidersDisabled() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.DisableKubeletCloudCredentialProviders)
}

// NewDockerKeyring creates a DockerKeyring to use for resolving credentials,
// which draws from the set of registered credential providers.
func NewDockerKeyring() DockerKeyring {
	keyring := &providersDockerKeyring{
		Providers: make([]DockerConfigProvider, 0),
	}

	keys := reflect.ValueOf(providers).MapKeys()
	stringKeys := make([]string, len(keys))
	for ix := range keys {
		stringKeys[ix] = keys[ix].String()
	}
	sort.Strings(stringKeys)

	for _, key := range stringKeys {
		provider := providers[key]
		if provider.Enabled() {
			klog.V(4).Infof("Registering credential provider: %v", key)
			keyring.Providers = append(keyring.Providers, provider)
		}
	}

	return keyring
}
