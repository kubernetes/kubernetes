/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeProtectedAttributes implements ProtectedAttributeInterface
type FakeProtectedAttributes struct {
	Fake *FakeRbac
	ns   string
}

var protectedattributesResource = unversioned.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "", Resource: "protectedattributes"}

func (c *FakeProtectedAttributes) Create(protectedAttribute *rbac.ProtectedAttribute) (result *rbac.ProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(protectedattributesResource, c.ns, protectedAttribute), &rbac.ProtectedAttribute{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttribute), err
}

func (c *FakeProtectedAttributes) Update(protectedAttribute *rbac.ProtectedAttribute) (result *rbac.ProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(protectedattributesResource, c.ns, protectedAttribute), &rbac.ProtectedAttribute{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttribute), err
}

func (c *FakeProtectedAttributes) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(protectedattributesResource, c.ns, name), &rbac.ProtectedAttribute{})

	return err
}

func (c *FakeProtectedAttributes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(protectedattributesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &rbac.ProtectedAttributeList{})
	return err
}

func (c *FakeProtectedAttributes) Get(name string) (result *rbac.ProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(protectedattributesResource, c.ns, name), &rbac.ProtectedAttribute{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttribute), err
}

func (c *FakeProtectedAttributes) List(opts api.ListOptions) (result *rbac.ProtectedAttributeList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(protectedattributesResource, c.ns, opts), &rbac.ProtectedAttributeList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &rbac.ProtectedAttributeList{}
	for _, item := range obj.(*rbac.ProtectedAttributeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested protectedAttributes.
func (c *FakeProtectedAttributes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(protectedattributesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched protectedAttribute.
func (c *FakeProtectedAttributes) Patch(name string, pt api.PatchType, data []byte) (result *rbac.ProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchAction(protectedattributesResource, c.ns, name, data), &rbac.ProtectedAttribute{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttribute), err
}
