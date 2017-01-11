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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeClusterRoles implements ClusterRoleInterface
type FakeClusterRoles struct {
	Fake *FakeRbacV1alpha1
}

var clusterrolesResource = schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "clusterroles"}

func (c *FakeClusterRoles) Create(clusterRole *v1alpha1.ClusterRole) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(clusterrolesResource, clusterRole), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) Update(clusterRole *v1alpha1.ClusterRole) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(clusterrolesResource, clusterRole), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(clusterrolesResource, name), &v1alpha1.ClusterRole{})
	return err
}

func (c *FakeClusterRoles) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(clusterrolesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.ClusterRoleList{})
	return err
}

func (c *FakeClusterRoles) Get(name string, options meta_v1.GetOptions) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(clusterrolesResource, name), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) List(opts v1.ListOptions) (result *v1alpha1.ClusterRoleList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(clusterrolesResource, opts), &v1alpha1.ClusterRoleList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.ClusterRoleList{}
	for _, item := range obj.(*v1alpha1.ClusterRoleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusterRoles.
func (c *FakeClusterRoles) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(clusterrolesResource, opts))
}

// Patch applies the patch and returns the patched clusterRole.
func (c *FakeClusterRoles) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(clusterrolesResource, name, data, subresources...), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}
