/*
Copyright 2017 The Kubernetes Authors.

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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeClusterRoles implements ClusterRoleInterface
type FakeClusterRoles struct {
	Fake *FakeRbac
}

var clusterrolesResource = schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "", Resource: "clusterroles"}

func (c *FakeClusterRoles) Create(clusterRole *rbac.ClusterRole) (result *rbac.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(clusterrolesResource, clusterRole), &rbac.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) Update(clusterRole *rbac.ClusterRole) (result *rbac.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(clusterrolesResource, clusterRole), &rbac.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(clusterrolesResource, name), &rbac.ClusterRole{})
	return err
}

func (c *FakeClusterRoles) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(clusterrolesResource, listOptions)

	_, err := c.Fake.Invokes(action, &rbac.ClusterRoleList{})
	return err
}

func (c *FakeClusterRoles) Get(name string, options v1.GetOptions) (result *rbac.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(clusterrolesResource, name), &rbac.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRole), err
}

func (c *FakeClusterRoles) List(opts api.ListOptions) (result *rbac.ClusterRoleList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(clusterrolesResource, opts), &rbac.ClusterRoleList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &rbac.ClusterRoleList{}
	for _, item := range obj.(*rbac.ClusterRoleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusterRoles.
func (c *FakeClusterRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(clusterrolesResource, opts))
}

// Patch applies the patch and returns the patched clusterRole.
func (c *FakeClusterRoles) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(clusterrolesResource, name, data, subresources...), &rbac.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRole), err
}
