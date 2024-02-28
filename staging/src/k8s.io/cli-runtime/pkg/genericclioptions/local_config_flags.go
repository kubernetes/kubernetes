/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// LocalClientConfig is a wrapper to catch empty config errors
// and return valid config to support local runs without accessing to API Server.
type LocalClientConfig struct {
	delegate clientcmd.ClientConfig
}

func NewLocalClientConfig(deletegate clientcmd.ClientConfig) clientcmd.ClientConfig {
	return &LocalClientConfig{
		delegate: deletegate,
	}
}

func (c *LocalClientConfig) Namespace() (string, bool, error) {
	ns, explicit, err := c.delegate.Namespace()
	if err != nil && clientcmd.IsEmptyConfig(err) {
		return "default", false, nil
	}
	return ns, explicit, err
}

func (c *LocalClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return c.delegate.RawConfig()
}
func (c *LocalClientConfig) ClientConfig() (*rest.Config, error) {
	config, err := c.delegate.ClientConfig()
	if err != nil && clientcmd.IsEmptyConfig(err) {
		return &rest.Config{}, nil
	}
	return config, err
}
func (c *LocalClientConfig) ConfigAccess() clientcmd.ConfigAccess {
	return c.delegate.ConfigAccess()
}
