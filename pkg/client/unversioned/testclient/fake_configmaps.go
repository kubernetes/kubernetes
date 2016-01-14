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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	configMapResourceName string = "configMaps"
)

// FakeConfigMaps implements ConfigMapInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeConfigMaps struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeConfigMaps) Get(name string) (*extensions.ConfigMap, error) {
	obj, err := c.Fake.Invokes(NewGetAction(configMapResourceName, c.Namespace, name), &extensions.ConfigMap{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ConfigMap), err
}

func (c *FakeConfigMaps) List(opts api.ListOptions) (*extensions.ConfigMapList, error) {
	obj, err := c.Fake.Invokes(NewListAction(configMapResourceName, c.Namespace, opts), &extensions.ConfigMapList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ConfigMapList), err
}

func (c *FakeConfigMaps) Create(cfg *extensions.ConfigMap) (*extensions.ConfigMap, error) {
	obj, err := c.Fake.Invokes(NewCreateAction(configMapResourceName, c.Namespace, cfg), cfg)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ConfigMap), err
}

func (c *FakeConfigMaps) Update(cfg *extensions.ConfigMap) (*extensions.ConfigMap, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction(configMapResourceName, c.Namespace, cfg), cfg)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ConfigMap), err
}

func (c *FakeConfigMaps) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction(configMapResourceName, c.Namespace, name), &extensions.ConfigMap{})
	return err
}

func (c *FakeConfigMaps) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction(configMapResourceName, c.Namespace, opts))
}
