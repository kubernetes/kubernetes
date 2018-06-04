/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type TestConfigFlags struct {
	clientConfig    clientcmd.ClientConfig
	discoveryClient discovery.CachedDiscoveryInterface
	restMapper      meta.RESTMapper
}

func (f *TestConfigFlags) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	if f.clientConfig == nil {
		panic("attempt to obtain a test RawKubeConfigLoader with no clientConfig specified")
	}
	return f.clientConfig
}

func (f *TestConfigFlags) ToRESTConfig() (*rest.Config, error) {
	return f.ToRawKubeConfigLoader().ClientConfig()
}

func (f *TestConfigFlags) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return f.discoveryClient, nil
}

func (f *TestConfigFlags) ToRESTMapper() (meta.RESTMapper, error) {
	if f.restMapper != nil {
		return f.restMapper, nil
	}
	if f.discoveryClient != nil {
		mapper := restmapper.NewDeferredDiscoveryRESTMapper(f.discoveryClient)
		expander := restmapper.NewShortcutExpander(mapper, f.discoveryClient)
		return expander, nil
	}
	return nil, fmt.Errorf("no restmapper")
}

func (f *TestConfigFlags) WithClientConfig(clientConfig clientcmd.ClientConfig) *TestConfigFlags {
	f.clientConfig = clientConfig
	return f
}

func (f *TestConfigFlags) WithRESTMapper(mapper meta.RESTMapper) *TestConfigFlags {
	f.restMapper = mapper
	return f
}

func (f *TestConfigFlags) WithDiscoveryClient(c discovery.CachedDiscoveryInterface) *TestConfigFlags {
	f.discoveryClient = c
	return f
}

func (f *TestConfigFlags) WithNamespace(ns string) *TestConfigFlags {
	if f.clientConfig == nil {
		panic("attempt to obtain a test RawKubeConfigLoader with no clientConfig specified")
	}
	f.clientConfig = &namespacedClientConfig{
		delegate:  f.clientConfig,
		namespace: ns,
	}
	return f
}

func NewTestConfigFlags() *TestConfigFlags {
	return &TestConfigFlags{}
}

type namespacedClientConfig struct {
	delegate  clientcmd.ClientConfig
	namespace string
}

func (c *namespacedClientConfig) Namespace() (string, bool, error) {
	return c.namespace, false, nil
}

func (c *namespacedClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return c.delegate.RawConfig()
}
func (c *namespacedClientConfig) ClientConfig() (*rest.Config, error) {
	return c.delegate.ClientConfig()
}
func (c *namespacedClientConfig) ConfigAccess() clientcmd.ConfigAccess {
	return c.delegate.ConfigAccess()
}
