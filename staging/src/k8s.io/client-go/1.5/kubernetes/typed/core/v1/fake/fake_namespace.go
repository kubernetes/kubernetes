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
	api "k8s.io/client-go/1.5/pkg/api"
	unversioned "k8s.io/client-go/1.5/pkg/api/unversioned"
	v1 "k8s.io/client-go/1.5/pkg/api/v1"
	labels "k8s.io/client-go/1.5/pkg/labels"
	watch "k8s.io/client-go/1.5/pkg/watch"
	testing "k8s.io/client-go/1.5/testing"
)

// FakeNamespaces implements NamespaceInterface
type FakeNamespaces struct {
	Fake *FakeCore
}

var namespacesResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}

func (c *FakeNamespaces) Create(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(namespacesResource, namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) Update(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(namespacesResource, namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) UpdateStatus(namespace *v1.Namespace) (*v1.Namespace, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(namespacesResource, "status", namespace), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(namespacesResource, name), &v1.Namespace{})
	return err
}

func (c *FakeNamespaces) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(namespacesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1.NamespaceList{})
	return err
}

func (c *FakeNamespaces) Get(name string) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(namespacesResource, name), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}

func (c *FakeNamespaces) List(opts api.ListOptions) (result *v1.NamespaceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(namespacesResource, opts), &v1.NamespaceList{})
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
		InvokesWatch(testing.NewRootWatchAction(namespacesResource, opts))
}

// Patch applies the patch and returns the patched namespace.
func (c *FakeNamespaces) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Namespace, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(namespacesResource, name, data, subresources...), &v1.Namespace{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Namespace), err
}
