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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeRoleBindings implements RoleBindingInterface
type FakeRoleBindings struct {
	Fake      *FakeRbac
	Namespace string
}

func (c *FakeRoleBindings) Get(name string) (*rbac.RoleBinding, error) {
	obj, err := c.Fake.Invokes(NewGetAction("rolebindings", c.Namespace, name), &rbac.RoleBinding{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.RoleBinding), err
}

func (c *FakeRoleBindings) List(opts api.ListOptions) (*rbac.RoleBindingList, error) {
	obj, err := c.Fake.Invokes(NewListAction("rolebindings", c.Namespace, opts), &rbac.RoleBindingList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.RoleBindingList), err
}

func (c *FakeRoleBindings) Create(csr *rbac.RoleBinding) (*rbac.RoleBinding, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("rolebindings", c.Namespace, csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.RoleBinding), err
}

func (c *FakeRoleBindings) Update(csr *rbac.RoleBinding) (*rbac.RoleBinding, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("rolebindings", c.Namespace, csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.RoleBinding), err
}

func (c *FakeRoleBindings) Delete(name string, opts *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("rolebindings", c.Namespace, name), &rbac.RoleBinding{})
	return err
}

func (c *FakeRoleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("rolebindings", c.Namespace, opts))
}
