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
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeClusterRoleBindings implements ClusterRoleBindingInterface
type FakeClusterRoleBindings struct {
	Fake *FakeRbac
}

var clusterrolebindingsResource = unversioned.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "", Resource: "clusterrolebindings"}

func (c *FakeClusterRoleBindings) Create(clusterRoleBinding *rbac.ClusterRoleBinding) (result *rbac.ClusterRoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(clusterrolebindingsResource, clusterRoleBinding), &rbac.ClusterRoleBinding{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) Update(clusterRoleBinding *rbac.ClusterRoleBinding) (result *rbac.ClusterRoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(clusterrolebindingsResource, clusterRoleBinding), &rbac.ClusterRoleBinding{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(clusterrolebindingsResource, name), &rbac.ClusterRoleBinding{})
	return err
}

func (c *FakeClusterRoleBindings) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(clusterrolebindingsResource, listOptions)

	_, err := c.Fake.Invokes(action, &rbac.ClusterRoleBindingList{})
	return err
}

func (c *FakeClusterRoleBindings) Get(name string) (result *rbac.ClusterRoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(clusterrolebindingsResource, name), &rbac.ClusterRoleBinding{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRoleBinding), err
}

func (c *FakeClusterRoleBindings) List(opts api.ListOptions) (result *rbac.ClusterRoleBindingList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(clusterrolebindingsResource, opts), &rbac.ClusterRoleBindingList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &rbac.ClusterRoleBindingList{}
	for _, item := range obj.(*rbac.ClusterRoleBindingList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusterRoleBindings.
func (c *FakeClusterRoleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(clusterrolebindingsResource, opts))
}

// Patch applies the patch and returns the patched clusterRoleBinding.
func (c *FakeClusterRoleBindings) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.ClusterRoleBinding, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(clusterrolebindingsResource, name, data, subresources...), &rbac.ClusterRoleBinding{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRoleBinding), err
}
