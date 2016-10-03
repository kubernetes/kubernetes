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

// FakeClusterRoles implements ClusterRoleInterface
type FakeClusterRoles struct {
	Fake *FakeRbac
}

var clusterrolesResource = unversioned.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "clusterroles"}

func (c *FakeClusterRoles) Create(clusterRole *v1alpha1.ClusterRole) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(clusterrolesResource, clusterRole), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) Update(clusterRole *v1alpha1.ClusterRole) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(clusterrolesResource, clusterRole), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(clusterrolesResource, name), &v1alpha1.ClusterRole{})
	return err
}

func (c *FakeClusterRoles) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(clusterrolesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.ClusterRoleList{})
	return err
}

func (c *FakeClusterRoles) Get(name string) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(clusterrolesResource, name), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}

func (c *FakeClusterRoles) List(opts api.ListOptions) (result *v1alpha1.ClusterRoleList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(clusterrolesResource, opts), &v1alpha1.ClusterRoleList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
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
func (c *FakeClusterRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(clusterrolesResource, opts))
}

// Patch applies the patch and returns the patched clusterRole.
func (c *FakeClusterRoles) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(clusterrolesResource, name, data, subresources...), &v1alpha1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ClusterRole), err
}
