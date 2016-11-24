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

package v1

import (
	fmt "fmt"
	api "k8s.io/kubernetes/pkg/api"
	registered "k8s.io/kubernetes/pkg/apimachinery/registered"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	serializer "k8s.io/kubernetes/pkg/runtime/serializer"
)

type CoreV1Interface interface {
	RESTClient() restclient.Interface
	ConfigMapsGetter
	EventsGetter
	NamespacesGetter
	SecretsGetter
	ServicesGetter
}

// CoreV1Client is used to interact with features provided by the k8s.io/kubernetes/pkg/apimachinery/registered.Group group.
type CoreV1Client struct {
	restClient restclient.Interface
}

func (c *CoreV1Client) ConfigMaps(namespace string) ConfigMapInterface {
	return newConfigMaps(c, namespace)
}

func (c *CoreV1Client) Events(namespace string) EventInterface {
	return newEvents(c, namespace)
}

func (c *CoreV1Client) Namespaces() NamespaceInterface {
	return newNamespaces(c)
}

func (c *CoreV1Client) Secrets(namespace string) SecretInterface {
	return newSecrets(c, namespace)
}

func (c *CoreV1Client) Services(namespace string) ServiceInterface {
	return newServices(c, namespace)
}

// NewForConfig creates a new CoreV1Client for the given config.
func NewForConfig(c *restclient.Config) (*CoreV1Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &CoreV1Client{client}, nil
}

// NewForConfigOrDie creates a new CoreV1Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *CoreV1Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new CoreV1Client for the given RESTClient.
func New(c restclient.Interface) *CoreV1Client {
	return &CoreV1Client{c}
}

func setConfigDefaults(config *restclient.Config) error {
	gv, err := schema.ParseGroupVersion("/v1")
	if err != nil {
		return err
	}
	// if /v1 is not enabled, return an error
	if !registered.IsEnabledVersion(gv) {
		return fmt.Errorf("/v1 is not enabled")
	}
	config.APIPath = "/api"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	copyGroupVersion := gv
	config.GroupVersion = &copyGroupVersion

	config.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: api.Codecs}

	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *CoreV1Client) RESTClient() restclient.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
