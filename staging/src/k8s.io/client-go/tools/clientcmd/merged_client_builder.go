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

	"k8s.io/api/core/v1"
	restclient "k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
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

	// provided for testing
	icc InClusterConfig
}

// InClusterConfig abstracts details of whether the client is running in a cluster for testing.
type InClusterConfig interface {
	ClientConfig
	Possible() bool
}

// NewNonInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name
func NewNonInteractiveDeferredLoadingClientConfig(loader ClientConfigLoader, overrides *ConfigOverrides) ClientConfig {
	return &DeferredLoadingClientConfig{loader: loader, overrides: overrides, icc: &inClusterClientConfig{overrides: overrides}}
}

// NewInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name and the fallback auth reader
func NewInteractiveDeferredLoadingClientConfig(loader ClientConfigLoader, overrides *ConfigOverrides, fallbackReader io.Reader) ClientConfig {
	return &DeferredLoadingClientConfig{loader: loader, overrides: overrides, icc: &inClusterClientConfig{overrides: overrides}, fallbackReader: fallbackReader}
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

	// load the configuration and return on non-empty errors and if the
	// content differs from the default config
	mergedConfig, err := mergedClientConfig.ClientConfig()
	if err != nil {
		if !IsEmptyConfig(err) {
			// return on any error except empty config
			return nil, err
		}
	}
	// the configuration is valid, but if this is equal to the defaults we should try
	// in-cluster configuration
	if config.isInClusterConfig(mergedConfig) {
		glog.V(4).Infof("Using in-cluster configuration")
		return config.icc.ClientConfig()
	}
	return mergedConfig, err
}

func (config *DeferredLoadingClientConfig) isInClusterConfig(mergedConfig *restclient.Config) bool {
	if !config.icc.Possible() {
		return false
	}
	if mergedConfig == nil {
		return true
	}

	defaultClientConfig, err := DefaultClientConfig.ClientConfig()
	if err != nil {
		return false
	}
	// compare with the DefaultClientConfig, but we ignore the following command line overrides:
	// --request-timeout, --as, --as-group.
	// If the mergedConfig differs from the DefaultClientConfig only in these fields, that means
	// user want to apply these overrides to in-cluster config.
	if len(config.overrides.Timeout) > 0 {
		timeout, err := ParseTimeout(config.overrides.Timeout)
		if err != nil {
			return false
		}
		defaultClientConfig.Timeout = timeout
	}
	defaultClientConfig.Impersonate = mergedConfig.Impersonate
	// compare all other parts, if any of them specified, use mergedConfig.
	return reflect.DeepEqual(mergedConfig, defaultClientConfig)
}

// Namespace implements KubeConfig
func (config *DeferredLoadingClientConfig) Namespace() (string, bool, error) {
	mergedKubeConfig, err := config.createClientConfig()
	if err != nil {
		return "", false, err
	}

	ns, overridden, err := mergedKubeConfig.Namespace()
	// if we get an error and it is not empty config, or if the merged config defined an explicit namespace, or
	// if in-cluster config is not possible, return immediately
	if (err != nil && !IsEmptyConfig(err)) || overridden || !config.icc.Possible() {
		// return on any error except empty config
		return ns, overridden, err
	}

	if len(ns) > 0 {
		// if we got a non-default namespace from the kubeconfig, use it
		if ns != v1.NamespaceDefault {
			return ns, false, nil
		}

		// if we got a default namespace, determine whether it was explicit or implicit
		if raw, err := mergedKubeConfig.RawConfig(); err == nil {
			if context := raw.Contexts[raw.CurrentContext]; context != nil && len(context.Namespace) > 0 {
				return ns, false, nil
			}
		}
	}

	glog.V(4).Infof("Using in-cluster namespace")

	// allow the namespace from the service account token directory to be used.
	return config.icc.Namespace()
}

// ConfigAccess implements ClientConfig
func (config *DeferredLoadingClientConfig) ConfigAccess() ConfigAccess {
	return config.loader
}
