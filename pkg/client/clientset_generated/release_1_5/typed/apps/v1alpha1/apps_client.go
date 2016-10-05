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

package v1alpha1

import (
	api "k8s.io/kubernetes/pkg/api"
	registered "k8s.io/kubernetes/pkg/apimachinery/registered"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	serializer "k8s.io/kubernetes/pkg/runtime/serializer"
)

type AppsInterface interface {
	GetRESTClient() restclient.RESTClientInterface
	PetSetsGetter
}

// AppsClient is used to interact with features provided by the Apps group.
type AppsClient struct {
	RESTClient restclient.RESTClientInterface
}

func (c *AppsClient) PetSets(namespace string) PetSetInterface {
	return newPetSets(c, namespace)
}

// NewForConfig creates a new AppsClient for the given config.
func NewForConfig(c *restclient.Config) (*AppsClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &AppsClient{client}, nil
}

// NewForConfigOrDie creates a new AppsClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *AppsClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new AppsClient for the given RESTClient.
func New(c restclient.RESTClientInterface) *AppsClient {
	return &AppsClient{c}
}

func setConfigDefaults(config *restclient.Config) error {
	// if apps group is not registered, return an error
	g, err := registered.Group("apps")
	if err != nil {
		return err
	}
	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	config.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: api.Codecs}

	return nil
}

// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *AppsClient) GetRESTClient() restclient.RESTClientInterface {
	if c == nil {
		return nil
	}
	return c.RESTClient
}
