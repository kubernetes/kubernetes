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
	api "k8s.io/client-go/1.4/pkg/api"
	unversioned "k8s.io/client-go/1.4/pkg/api/unversioned"
	v1alpha1 "k8s.io/client-go/1.4/pkg/apis/rbac/v1alpha1"
	labels "k8s.io/client-go/1.4/pkg/labels"
	watch "k8s.io/client-go/1.4/pkg/watch"
	testing "k8s.io/client-go/1.4/testing"
)

// FakeRoleBindings implements RoleBindingInterface
type FakeRoleBindings struct {
	Fake *FakeRbac
	ns   string
}

var rolebindingsResource = unversioned.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "rolebindings"}

func (c *FakeRoleBindings) Create(roleBinding *v1alpha1.RoleBinding) (result *v1alpha1.RoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(rolebindingsResource, c.ns, roleBinding), &v1alpha1.RoleBinding{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.RoleBinding), err
}

func (c *FakeRoleBindings) Update(roleBinding *v1alpha1.RoleBinding) (result *v1alpha1.RoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(rolebindingsResource, c.ns, roleBinding), &v1alpha1.RoleBinding{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.RoleBinding), err
}

func (c *FakeRoleBindings) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(rolebindingsResource, c.ns, name), &v1alpha1.RoleBinding{})

	return err
}

func (c *FakeRoleBindings) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(rolebindingsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.RoleBindingList{})
	return err
}

func (c *FakeRoleBindings) Get(name string) (result *v1alpha1.RoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(rolebindingsResource, c.ns, name), &v1alpha1.RoleBinding{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.RoleBinding), err
}

func (c *FakeRoleBindings) List(opts api.ListOptions) (result *v1alpha1.RoleBindingList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(rolebindingsResource, c.ns, opts), &v1alpha1.RoleBindingList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.RoleBindingList{}
	for _, item := range obj.(*v1alpha1.RoleBindingList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested roleBindings.
func (c *FakeRoleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(rolebindingsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched roleBinding.
func (c *FakeRoleBindings) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.RoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(rolebindingsResource, c.ns, name, data, subresources...), &v1alpha1.RoleBinding{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.RoleBinding), err
}
