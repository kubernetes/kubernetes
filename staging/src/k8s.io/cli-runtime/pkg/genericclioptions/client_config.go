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

package genericclioptions

import (
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

var (
	ErrEmptyConfig = clientcmd.NewEmptyConfigError(`Missing or incomplete configuration info.  Please point to an existing, complete config file:


  1. Via the command-line flag --kubeconfig
  2. Via the KUBECONFIG environment variable
  3. In your home directory as ~/.kube/config

To view or setup config directly use the 'config' command.`)
)

var _ = clientcmd.ClientConfig(&clientConfig{})

type clientConfig struct {
	defaultClientConfig clientcmd.ClientConfig
}

func (c *clientConfig) RawConfig() (clientcmdapi.Config, error) {
	config, err := c.defaultClientConfig.RawConfig()
	// replace client-go's ErrEmptyConfig error with our custom, more verbose version
	if clientcmd.IsEmptyConfig(err) {
		return config, ErrEmptyConfig
	}
	return config, err
}

func (c *clientConfig) ClientConfig() (*restclient.Config, error) {
	config, err := c.defaultClientConfig.ClientConfig()
	// replace client-go's ErrEmptyConfig error with our custom, more verbose version
	if clientcmd.IsEmptyConfig(err) {
		return config, ErrEmptyConfig
	}
	return config, err
}

func (c *clientConfig) Namespace() (string, bool, error) {
	namespace, ok, err := c.defaultClientConfig.Namespace()
	// replace client-go's ErrEmptyConfig error with our custom, more verbose version
	if clientcmd.IsEmptyConfig(err) {
		return namespace, ok, ErrEmptyConfig
	}
	return namespace, ok, err
}

func (c *clientConfig) ConfigAccess() clientcmd.ConfigAccess {
	return c.defaultClientConfig.ConfigAccess()
}
