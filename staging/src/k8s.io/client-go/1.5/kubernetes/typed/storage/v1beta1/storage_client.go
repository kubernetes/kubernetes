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

package v1beta1

import (
	api "k8s.io/client-go/1.5/pkg/api"
	registered "k8s.io/client-go/1.5/pkg/apimachinery/registered"
	serializer "k8s.io/client-go/1.5/pkg/runtime/serializer"
	rest "k8s.io/client-go/1.5/rest"
)

type StorageInterface interface {
	GetRESTClient() *rest.RESTClient
	StorageClassesGetter
}

// StorageClient is used to interact with features provided by the Storage group.
type StorageClient struct {
	*rest.RESTClient
}

func (c *StorageClient) StorageClasses() StorageClassInterface {
	return newStorageClasses(c)
}

// NewForConfig creates a new StorageClient for the given config.
func NewForConfig(c *rest.Config) (*StorageClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &StorageClient{client}, nil
}

// NewForConfigOrDie creates a new StorageClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *StorageClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new StorageClient for the given RESTClient.
func New(c *rest.RESTClient) *StorageClient {
	return &StorageClient{c}
}

func setConfigDefaults(config *rest.Config) error {
	// if storage group is not registered, return an error
	g, err := registered.Group("storage.k8s.io")
	if err != nil {
		return err
	}
	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
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
func (c *StorageClient) GetRESTClient() *rest.RESTClient {
	if c == nil {
		return nil
	}
	return c.RESTClient
}
