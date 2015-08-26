/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package networkprovider

import (
	"fmt"
	"sync"

	"github.com/golang/glog"
)

// Factory is a function that returns a networkprovider.Interface.
// The config parameter provides an io.Reader handler to the factory in
// order to load specific configurations. If no configuration is provided
// the parameter is nil.
type Factory func() (Interface, error)

// All registered network providers.
var providersMutex sync.Mutex
var providers = make(map[string]Factory)

// RegisterNetworkProvider registers a networkprovider.Factory by name.  This
// is expected to happen during app startup.
func RegisterNetworkProvider(name string, networkProvider Factory) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	if _, found := providers[name]; found {
		glog.Fatalf("Network provider %q was registered twice", name)
	}
	glog.V(1).Infof("Registered network provider %q", name)
	providers[name] = networkProvider
}

// GetNetworkProvider creates an instance of the named network provider, or nil if
// the name is not known.  The error return is only used if the named provider
// was known but failed to initialize.
func GetNetworkProvider(name string) (Interface, error) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	f, found := providers[name]
	if !found {
		return nil, nil
	}
	return f()
}

// InitNetworkProvider creates an instance of the named networkProvider provider.
func InitNetworkProvider(name string) (Interface, error) {
	var networkProvider Interface

	if name == "" {
		glog.Info("No network provider specified.")
		return nil, nil
	}

	var err error
	networkProvider, err = GetNetworkProvider(name)

	if err != nil {
		return nil, fmt.Errorf("could not init networkProvider provider %q: %v", name, err)
	}
	if networkProvider == nil {
		return nil, fmt.Errorf("unknown networkProvider provider %q", name)
	}

	return networkProvider, nil
}
