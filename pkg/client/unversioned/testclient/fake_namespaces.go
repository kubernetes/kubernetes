/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeNamespaces implements NamespacesInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeNamespaces struct {
	Fake *Fake
}

func (c *FakeNamespaces) Get(name string) (*api.Namespace, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("namespaces", name), &api.Namespace{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) List(label labels.Selector, field fields.Selector) (*api.NamespaceList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("namespaces", label, field), &api.NamespaceList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.NamespaceList), err
}

func (c *FakeNamespaces) Create(namespace *api.Namespace) (*api.Namespace, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("namespaces", namespace), namespace)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Namespace), c.Fake.Err()
}

func (c *FakeNamespaces) Update(namespace *api.Namespace) (*api.Namespace, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("namespaces", namespace), namespace)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("namespaces", name), &api.Namespace{})
	return err
}

func (c *FakeNamespaces) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(NewRootWatchAction("namespaces", label, field, resourceVersion), nil)
	return c.Fake.Watch, nil
}

func (c *FakeNamespaces) Finalize(namespace *api.Namespace) (*api.Namespace, error) {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "namespaces"
	action.Subresource = "finalize"
	action.Object = namespace

	obj, err := c.Fake.Invokes(action, namespace)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) Status(namespace *api.Namespace) (*api.Namespace, error) {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "namespaces"
	action.Subresource = "status"
	action.Object = namespace

	obj, err := c.Fake.Invokes(action, namespace)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Namespace), err
}
