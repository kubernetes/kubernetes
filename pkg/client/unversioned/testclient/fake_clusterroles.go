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

// FakeClusterRoles implements ClusterRoleInterface
type FakeClusterRoles struct {
	Fake *FakeRbac
}

func (c *FakeClusterRoles) Get(name string) (*rbac.ClusterRole, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("clusterroles", name), &rbac.ClusterRole{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) List(opts api.ListOptions) (*rbac.ClusterRoleList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("clusterroles", opts), &rbac.ClusterRoleList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleList), err
}

func (c *FakeClusterRoles) Create(csr *rbac.ClusterRole) (*rbac.ClusterRole, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("clusterroles", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) Update(csr *rbac.ClusterRole) (*rbac.ClusterRole, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("clusterroles", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) Delete(name string, opts *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("clusterroles", name), &rbac.ClusterRole{})
	return err
}

func (c *FakeClusterRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("clusterroles", opts))
}
