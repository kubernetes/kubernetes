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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/restclient"
)

type StorageInterface interface {
	StorageClassesInterface
}

// StorageClient is used to interact with Kubernetes storage features.
type StorageClient struct {
	*restclient.RESTClient
}

func (c *StorageClient) StorageClasses() StorageClassInterface {
	return newStorageClasses(c)
}

func NewStorage(c *restclient.Config) (*StorageClient, error) {
	config := *c
	if err := setStorageDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &StorageClient{client}, nil
}

func NewStorageOrDie(c *restclient.Config) *StorageClient {
	client, err := NewStorage(c)
	if err != nil {
		panic(err)
	}
	return client
}

func setStorageDefaults(config *restclient.Config) error {
	// if storage group is not registered, return an error
	g, err := registered.Group(storage.GroupName)
	if err != nil {
		return err
	}
	config.APIPath = defaultAPIPath
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	config.NegotiatedSerializer = api.Codecs
	return nil
}
