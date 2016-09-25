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

// FakeRoles implements RoleInterface
type FakeRoles struct {
	Fake      *FakeRbac
	Namespace string
}

func (c *FakeRoles) Get(name string) (*rbac.Role, error) {
	obj, err := c.Fake.Invokes(NewGetAction("roles", c.Namespace, name), &rbac.Role{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.Role), err
}

func (c *FakeRoles) List(opts api.ListOptions) (*rbac.RoleList, error) {
	obj, err := c.Fake.Invokes(NewListAction("roles", c.Namespace, opts), &rbac.RoleList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.RoleList), err
}

func (c *FakeRoles) Create(csr *rbac.Role) (*rbac.Role, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("roles", c.Namespace, csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.Role), err
}

func (c *FakeRoles) Update(csr *rbac.Role) (*rbac.Role, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("roles", c.Namespace, csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.Role), err
}

func (c *FakeRoles) Delete(name string, opts *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("roles", c.Namespace, name), &rbac.Role{})
	return err
}

func (c *FakeRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("roles", c.Namespace, opts))
}
