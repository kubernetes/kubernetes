/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	ConfigMapResourceName string = "configmaps"
)

type ConfigMapsNamespacer interface {
	ConfigMaps(namespace string) ConfigMapsInterface
}

type ConfigMapsInterface interface {
	Get(string) (*extensions.ConfigMap, error)
	List(opts api.ListOptions) (*extensions.ConfigMapList, error)
	Create(*extensions.ConfigMap) (*extensions.ConfigMap, error)
	Delete(string) error
	Update(*extensions.ConfigMap) (*extensions.ConfigMap, error)
	Watch(api.ListOptions) (watch.Interface, error)
}

type ConfigMaps struct {
	client    *ExtensionsClient
	namespace string
}

// ConfigMaps should implement ConfigMapsInterface
var _ ConfigMapsInterface = &ConfigMaps{}

func newConfigMaps(c *ExtensionsClient, ns string) *ConfigMaps {
	return &ConfigMaps{
		client:    c,
		namespace: ns,
	}
}

func (c *ConfigMaps) Get(name string) (*extensions.ConfigMap, error) {
	result := &extensions.ConfigMap{}
	err := c.client.Get().
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		Name(name).
		Do().
		Into(result)

	return result, err
}

func (c *ConfigMaps) List(opts api.ListOptions) (*extensions.ConfigMapList, error) {
	result := &extensions.ConfigMapList{}
	err := c.client.Get().
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)

	return result, err
}

func (c *ConfigMaps) Create(cfg *extensions.ConfigMap) (*extensions.ConfigMap, error) {
	result := &extensions.ConfigMap{}
	err := c.client.Post().
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		Body(cfg).
		Do().
		Into(result)

	return result, err
}

func (c *ConfigMaps) Delete(name string) error {
	return c.client.Delete().
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		Name(name).
		Do().
		Error()
}

func (c *ConfigMaps) Update(cfg *extensions.ConfigMap) (*extensions.ConfigMap, error) {
	result := &extensions.ConfigMap{}

	err := c.client.Put().
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		Name(cfg.Name).
		Body(cfg).
		Do().
		Into(result)

	return result, err
}

func (c *ConfigMaps) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.namespace).
		Resource(ConfigMapResourceName).
		VersionedParams(&opts, api.Scheme).
		Watch()
}
