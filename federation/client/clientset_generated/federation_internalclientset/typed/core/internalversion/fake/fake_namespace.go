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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeNamespaces implements NamespaceInterface
type FakeNamespaces struct {
	Fake *FakeCore
}

var namespacesResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "namespaces"}

func (c *FakeNamespaces) Create(namespace *api.Namespace) (result *api.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(namespacesResource, namespace), &api.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) Update(namespace *api.Namespace) (result *api.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(namespacesResource, namespace), &api.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) UpdateStatus(namespace *api.Namespace) (*api.Namespace, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(namespacesResource, "status", namespace), &api.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(namespacesResource, name), &api.Namespace{})
	return err
}

func (c *FakeNamespaces) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(namespacesResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.NamespaceList{})
	return err
}

func (c *FakeNamespaces) Get(name string) (result *api.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(namespacesResource, name), &api.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Namespace), err
}

func (c *FakeNamespaces) List(opts api.ListOptions) (result *api.NamespaceList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(namespacesResource, opts), &api.NamespaceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.NamespaceList{}
	for _, item := range obj.(*api.NamespaceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *FakeNamespaces) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(namespacesResource, opts))
}

// Patch applies the patch and returns the patched namespace.
func (c *FakeNamespaces) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(namespacesResource, name, data, subresources...), &api.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Namespace), err
}
