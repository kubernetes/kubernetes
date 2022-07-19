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

package kubeapiserver

import (
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	serveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	apisstorage "k8s.io/kubernetes/pkg/apis/storage"
)

// NewStorageFactoryConfig returns a new StorageFactoryConfig set up with necessary resource overrides.
func NewStorageFactoryConfig(scheme *runtime.Scheme, codecs serializer.CodecFactory) *StorageFactoryConfig {
	resources := []schema.GroupVersionResource{
		// TODO (https://github.com/kubernetes/kubernetes/issues/108451): remove the override in
		// 1.25.
		apisstorage.Resource("csistoragecapacities").WithVersion("v1beta1"),
	}
	return &StorageFactoryConfig{
		Serializer:                codecs,
		DefaultResourceEncoding:   serverstorage.NewDefaultResourceEncodingConfig(scheme),
		ResourceEncodingOverrides: resources,
	}
}

// StorageFactoryConfig is a configuration for creating storage factory.
type StorageFactoryConfig struct {
	StorageConfig                    storagebackend.Config
	APIResourceConfig                *serverstorage.ResourceConfig
	DefaultResourceEncoding          *serverstorage.DefaultResourceEncodingConfig
	DefaultStorageMediaType          string
	Serializer                       runtime.StorageSerializer
	ResourceEncodingOverrides        []schema.GroupVersionResource
	EtcdServersOverrides             []string
	EncryptionProviderConfigFilepath string
}

// Complete completes the StorageFactoryConfig with provided etcdOptions returning completedStorageFactoryConfig.
func (c *StorageFactoryConfig) Complete(etcdOptions *serveroptions.EtcdOptions) (*completedStorageFactoryConfig, error) {
	c.StorageConfig = etcdOptions.StorageConfig
	c.DefaultStorageMediaType = etcdOptions.DefaultStorageMediaType
	c.EtcdServersOverrides = etcdOptions.EtcdServersOverrides
	c.EncryptionProviderConfigFilepath = etcdOptions.EncryptionProviderConfigFilepath
	return &completedStorageFactoryConfig{c}, nil
}

// completedStorageFactoryConfig is a wrapper around StorageFactoryConfig completed with etcd options.
//
// Note: this struct is intentionally unexported so that it can only be constructed via a StorageFactoryConfig.Complete
// call. The implied consequence is that this does not comply with golint.
type completedStorageFactoryConfig struct {
	*StorageFactoryConfig
}

// New returns a new storage factory created from the completed storage factory configuration.
func (c *completedStorageFactoryConfig) New() (*serverstorage.DefaultStorageFactory, error) {
	resourceEncodingConfig := resourceconfig.MergeResourceEncodingConfigs(c.DefaultResourceEncoding, c.ResourceEncodingOverrides)
	storageFactory := serverstorage.NewDefaultStorageFactory(
		c.StorageConfig,
		c.DefaultStorageMediaType,
		c.Serializer,
		resourceEncodingConfig,
		c.APIResourceConfig,
		nil)

	for _, override := range c.EtcdServersOverrides {
		tokens := strings.Split(override, "#")
		apiresource := strings.Split(tokens[0], "/")

		group := apiresource[0]
		resource := apiresource[1]
		groupResource := schema.GroupResource{Group: group, Resource: resource}

		servers := strings.Split(tokens[1], ";")
		storageFactory.SetEtcdLocation(groupResource, servers)
	}
	if len(c.EncryptionProviderConfigFilepath) != 0 {
		transformerOverrides, err := encryptionconfig.GetTransformerOverrides(c.EncryptionProviderConfigFilepath)
		if err != nil {
			return nil, err
		}
		for groupResource, transformer := range transformerOverrides {
			storageFactory.SetTransformer(groupResource, transformer)
		}
	}
	return storageFactory, nil
}
