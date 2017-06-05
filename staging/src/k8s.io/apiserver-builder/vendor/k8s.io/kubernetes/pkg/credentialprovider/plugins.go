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

	"github.com/golang/glog"
)

// All registered credential providers.
var providersMutex sync.Mutex
var providers = make(map[string]DockerConfigProvider)

// RegisterCredentialProvider is called by provider implementations on
// initialization to register themselves, like so:
//   func init() {
//    	RegisterCredentialProvider("name", &myProvider{...})
//   }
func RegisterCredentialProvider(name string, provider DockerConfigProvider) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	_, found := providers[name]
	if found {
		glog.Fatalf("Credential provider %q was registered twice", name)
	}
	glog.V(4).Infof("Registered credential provider %q", name)
	providers[name] = provider
}

// NewDockerKeyring creates a DockerKeyring to use for resolving credentials,
// which lazily draws from the set of registered credential providers.
func NewDockerKeyring() DockerKeyring {
	keyring := &lazyDockerKeyring{
		Providers: make([]DockerConfigProvider, 0),
	}

	// TODO(mattmoor): iterating over the map is non-deterministic.  We should
	// introduce the notion of priorities for conflict resolution.
	for name, provider := range providers {
		if provider.Enabled() {
			glog.V(4).Infof("Registering credential provider: %v", name)
			keyring.Providers = append(keyring.Providers, provider)
		}
	}

	return keyring
}
