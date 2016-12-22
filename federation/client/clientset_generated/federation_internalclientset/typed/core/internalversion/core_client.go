/*
Copyright 2016 The Kubernetes Authors.

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

package internalversion

import (
	api "k8s.io/kubernetes/pkg/api"
	registered "k8s.io/kubernetes/pkg/apimachinery/registered"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

type CoreInterface interface {
	RESTClient() restclient.Interface
	ConfigMapsGetter
	EventsGetter
	NamespacesGetter
	SecretsGetter
	ServicesGetter
}

// CoreClient is used to interact with features provided by the k8s.io/kubernetes/pkg/apimachinery/registered.Group group.
type CoreClient struct {
	restClient restclient.Interface
}

func (c *CoreClient) ConfigMaps(namespace string) ConfigMapInterface {
	return newConfigMaps(c, namespace)
}

func (c *CoreClient) Events(namespace string) EventInterface {
	return newEvents(c, namespace)
}

func (c *CoreClient) Namespaces() NamespaceInterface {
	return newNamespaces(c)
}

func (c *CoreClient) Secrets(namespace string) SecretInterface {
	return newSecrets(c, namespace)
}

func (c *CoreClient) Services(namespace string) ServiceInterface {
	return newServices(c, namespace)
}

// NewForConfig creates a new CoreClient for the given config.
func NewForConfig(c *restclient.Config) (*CoreClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &CoreClient{client}, nil
}

// NewForConfigOrDie creates a new CoreClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *CoreClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new CoreClient for the given RESTClient.
func New(c restclient.Interface) *CoreClient {
	return &CoreClient{c}
}

func setConfigDefaults(config *restclient.Config) error {
	// if core group is not registered, return an error
	g, err := registered.Group("")
	if err != nil {
		return err
	}
	config.APIPath = "/api"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != g.GroupVersion.Group {
		copyGroupVersion := g.GroupVersion
		config.GroupVersion = &copyGroupVersion
	}
	config.NegotiatedSerializer = api.Codecs

	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *CoreClient) RESTClient() restclient.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
