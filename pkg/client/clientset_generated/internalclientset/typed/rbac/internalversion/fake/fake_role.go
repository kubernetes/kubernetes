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
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeRoles implements RoleInterface
type FakeRoles struct {
	Fake *FakeRbac
	ns   string
}

var rolesResource = schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "", Resource: "roles"}

func (c *FakeRoles) Create(role *rbac.Role) (result *rbac.Role, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(rolesResource, c.ns, role), &rbac.Role{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.Role), err
}

func (c *FakeRoles) Update(role *rbac.Role) (result *rbac.Role, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(rolesResource, c.ns, role), &rbac.Role{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.Role), err
}

func (c *FakeRoles) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(rolesResource, c.ns, name), &rbac.Role{})

	return err
}

func (c *FakeRoles) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(rolesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &rbac.RoleList{})
	return err
}

func (c *FakeRoles) Get(name string) (result *rbac.Role, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(rolesResource, c.ns, name), &rbac.Role{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.Role), err
}

func (c *FakeRoles) List(opts api.ListOptions) (result *rbac.RoleList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(rolesResource, c.ns, opts), &rbac.RoleList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &rbac.RoleList{}
	for _, item := range obj.(*rbac.RoleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested roles.
func (c *FakeRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(rolesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched role.
func (c *FakeRoles) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.Role, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(rolesResource, c.ns, name, data, subresources...), &rbac.Role{})

	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.Role), err
}
