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

package clientcmd

import (
	"io"
	"reflect"
	"sync"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/restclient"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

// DeferredLoadingClientConfig is a ClientConfig interface that is backed by a client config loader.
// It is used in cases where the loading rules may change after you've instantiated them and you want to be sure that
// the most recent rules are used.  This is useful in cases where you bind flags to loading rule parameters before
// the parse happens and you want your calling code to be ignorant of how the values are being mutated to avoid
// passing extraneous information down a call stack
type DeferredLoadingClientConfig struct {
	loader         ClientConfigLoader
	overrides      *ConfigOverrides
	fallbackReader io.Reader

	clientConfig ClientConfig
	loadingLock  sync.Mutex
}

// NewNonInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name
func NewNonInteractiveDeferredLoadingClientConfig(loader ClientConfigLoader, overrides *ConfigOverrides) ClientConfig {
	return &DeferredLoadingClientConfig{loader: loader, overrides: overrides}
}

// NewInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name and the fallback auth reader
func NewInteractiveDeferredLoadingClientConfig(loader ClientConfigLoader, overrides *ConfigOverrides, fallbackReader io.Reader) ClientConfig {
	return &DeferredLoadingClientConfig{loader: loader, overrides: overrides, fallbackReader: fallbackReader}
}

func (config *DeferredLoadingClientConfig) createClientConfig() (ClientConfig, error) {
	if config.clientConfig == nil {
		config.loadingLock.Lock()
		defer config.loadingLock.Unlock()

		if config.clientConfig == nil {
			mergedConfig, err := config.loader.Load()
			if err != nil {
				return nil, err
			}

			var mergedClientConfig ClientConfig
			if config.fallbackReader != nil {
				mergedClientConfig = NewInteractiveClientConfig(*mergedConfig, config.overrides.CurrentContext, config.overrides, config.fallbackReader, config.loader)
			} else {
				mergedClientConfig = NewNonInteractiveClientConfig(*mergedConfig, config.overrides.CurrentContext, config.overrides, config.loader)
			}

			config.clientConfig = mergedClientConfig
		}
	}

	return config.clientConfig, nil
}

func (config *DeferredLoadingClientConfig) RawConfig() (clientcmdapi.Config, error) {
	mergedConfig, err := config.createClientConfig()
	if err != nil {
		return clientcmdapi.Config{}, err
	}

	return mergedConfig.RawConfig()
}

// ClientConfig implements ClientConfig
func (config *DeferredLoadingClientConfig) ClientConfig() (*restclient.Config, error) {
	mergedClientConfig, err := config.createClientConfig()
	if err != nil {
		return nil, err
	}

	mergedConfig, err := mergedClientConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	// Are we running in a cluster and were no other configs found? If so, use the in-cluster-config.
	icc := inClusterClientConfig{}
	defaultConfig, err := DefaultClientConfig.ClientConfig()
	if icc.Possible() && err == nil && reflect.DeepEqual(mergedConfig, defaultConfig) {
		glog.V(2).Info("No kubeconfig could be created, falling back to service account.")
		return icc.ClientConfig()
	}
	return mergedConfig, nil
}

// Namespace implements KubeConfig
func (config *DeferredLoadingClientConfig) Namespace() (string, bool, error) {
	mergedKubeConfig, err := config.createClientConfig()
	if err != nil {
		return "", false, err
	}

	return mergedKubeConfig.Namespace()
}

// ConfigAccess implements ClientConfig
func (config *DeferredLoadingClientConfig) ConfigAccess() ConfigAccess {
	return config.loader
}
