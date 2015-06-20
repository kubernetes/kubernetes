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

package clientcmd

import (
	"io"
	"reflect"

	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

// DeferredLoadingClientConfig is a ClientConfig interface that is backed by a set of loading rules
// It is used in cases where the loading rules may change after you've instantiated them and you want to be sure that
// the most recent rules are used.  This is useful in cases where you bind flags to loading rule parameters before
// the parse happens and you want your calling code to be ignorant of how the values are being mutated to avoid
// passing extraneous information down a call stack
type DeferredLoadingClientConfig struct {
	loadingRules   *ClientConfigLoadingRules
	overrides      *ConfigOverrides
	fallbackReader io.Reader
}

// NewNonInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name
func NewNonInteractiveDeferredLoadingClientConfig(loadingRules *ClientConfigLoadingRules, overrides *ConfigOverrides) ClientConfig {
	return DeferredLoadingClientConfig{loadingRules, overrides, nil}
}

// NewInteractiveDeferredLoadingClientConfig creates a ConfigClientClientConfig using the passed context name and the fallback auth reader
func NewInteractiveDeferredLoadingClientConfig(loadingRules *ClientConfigLoadingRules, overrides *ConfigOverrides, fallbackReader io.Reader) ClientConfig {
	return DeferredLoadingClientConfig{loadingRules, overrides, fallbackReader}
}

func (config DeferredLoadingClientConfig) createClientConfig() (ClientConfig, error) {
	mergedConfig, err := config.loadingRules.Load()
	if err != nil {
		return nil, err
	}

	var mergedClientConfig ClientConfig
	if config.fallbackReader != nil {
		mergedClientConfig = NewInteractiveClientConfig(*mergedConfig, config.overrides.CurrentContext, config.overrides, config.fallbackReader)
	} else {
		mergedClientConfig = NewNonInteractiveClientConfig(*mergedConfig, config.overrides.CurrentContext, config.overrides)
	}

	return mergedClientConfig, nil
}

func (config DeferredLoadingClientConfig) RawConfig() (clientcmdapi.Config, error) {
	mergedConfig, err := config.createClientConfig()
	if err != nil {
		return clientcmdapi.Config{}, err
	}

	return mergedConfig.RawConfig()
}

// ClientConfig implements ClientConfig
func (config DeferredLoadingClientConfig) ClientConfig() (*client.Config, error) {
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
		glog.V(2).Info("no kubeconfig could be created, falling back to service account.")
		return icc.ClientConfig()
	}

	return mergedConfig, nil
}

// Namespace implements KubeConfig
func (config DeferredLoadingClientConfig) Namespace() (string, error) {
	mergedKubeConfig, err := config.createClientConfig()
	if err != nil {
		return "", err
	}

	return mergedKubeConfig.Namespace()
}
