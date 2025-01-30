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
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

type provider struct {
	name string
	impl DockerConfigProvider
}

// All registered credential providers.
var providersMutex sync.Mutex
var providers = make([]provider, 0)
var seenProviderNames = sets.NewString()

// RegisterCredentialProvider is called by provider implementations on
// initialization to register themselves, like so:
//
//	func init() {
//	 	RegisterCredentialProvider("name", &myProvider{...})
//	}
func RegisterCredentialProvider(name string, p DockerConfigProvider) {
	providersMutex.Lock()
	defer providersMutex.Unlock()

	if seenProviderNames.Has(name) {
		klog.Fatalf("Credential provider %q was registered twice", name)
	}
	seenProviderNames.Insert(name)

	providers = append(providers, provider{name, p})
	klog.V(4).Infof("Registered credential provider %q", name)
}

// NewDockerKeyring creates a DockerKeyring to use for resolving credentials,
// which draws from the set of registered credential providers.
func NewDockerKeyring() DockerKeyring {
	keyring := &providersDockerKeyring{
		Providers: make([]DockerConfigProvider, 0),
	}

	for _, p := range providers {
		if p.impl.Enabled() {
			klog.V(4).Infof("Registering credential provider: %v", p.name)
			keyring.Providers = append(keyring.Providers, p.impl)
		}
	}

	return keyring
}
