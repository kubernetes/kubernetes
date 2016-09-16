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
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeNamespaces implements NamespaceInterface
type FakeNamespaces struct {
	Fake *FakeCore
}

var namespacesResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}

func (c *FakeNamespaces) Create(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(namespacesResource, namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) Update(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(namespacesResource, namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) UpdateStatus(namespace *v1.Namespace) (*v1.Namespace, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(namespacesResource, "status", namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(namespacesResource, name), &v1.Namespace{})
	return err
}

func (c *FakeNamespaces) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(namespacesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1.NamespaceList{})
	return err
}

func (c *FakeNamespaces) Get(name string) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(namespacesResource, name), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) List(opts api.ListOptions) (result *v1.NamespaceList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(namespacesResource, opts), &v1.NamespaceList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.NamespaceList{}
	for _, item := range obj.(*v1.NamespaceList).Items {
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
func (c *FakeNamespaces) Patch(name string, pt api.PatchType, data []byte) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchAction(namespacesResource, name, data), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}
