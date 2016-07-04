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

// FakeClusterRoleBindings implements ClusterRoleBindingInterface
type FakeClusterRoleBindings struct {
	Fake *FakeRbac
}

func (c *FakeClusterRoleBindings) Get(name string) (*rbac.ClusterRoleBinding, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("clusterrolebindings", name), &rbac.ClusterRoleBinding{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) List(opts api.ListOptions) (*rbac.ClusterRoleBindingList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("clusterrolebindings", opts), &rbac.ClusterRoleBindingList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleBindingList), err
}

func (c *FakeClusterRoleBindings) Create(csr *rbac.ClusterRoleBinding) (*rbac.ClusterRoleBinding, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("clusterrolebindings", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) Update(csr *rbac.ClusterRoleBinding) (*rbac.ClusterRoleBinding, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("clusterrolebindings", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) Delete(name string, opts *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("clusterrolebindings", name), &rbac.ClusterRoleBinding{})
	return err
}

func (c *FakeClusterRoleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("clusterrolebindings", opts))
}
