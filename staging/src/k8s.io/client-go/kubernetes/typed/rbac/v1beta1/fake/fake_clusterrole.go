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
	v1beta1 "k8s.io/api/rbac/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeClusterRoles implements ClusterRoleInterface
type FakeClusterRoles struct {
	Fake *FakeRbacV1beta1
}

var clusterrolesResource = schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "clusterroles"}

var clusterrolesKind = schema.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Kind: "ClusterRole"}

// Get takes name of the clusterRole, and returns the corresponding clusterRole object, and an error if there is any.
func (c *FakeClusterRoles) Get(name string, options v1.GetOptions) (result *v1beta1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(clusterrolesResource, name), &v1beta1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ClusterRole), err
}

// List takes label and field selectors, and returns the list of ClusterRoles that match those selectors.
func (c *FakeClusterRoles) List(opts v1.ListOptions) (result *v1beta1.ClusterRoleList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(clusterrolesResource, clusterrolesKind, opts), &v1beta1.ClusterRoleList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.ClusterRoleList{}
	for _, item := range obj.(*v1beta1.ClusterRoleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusterRoles.
func (c *FakeClusterRoles) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(clusterrolesResource, opts))
}

// Create takes the representation of a clusterRole and creates it.  Returns the server's representation of the clusterRole, and an error, if there is any.
func (c *FakeClusterRoles) Create(clusterRole *v1beta1.ClusterRole) (result *v1beta1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(clusterrolesResource, clusterRole), &v1beta1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ClusterRole), err
}

// Update takes the representation of a clusterRole and updates it. Returns the server's representation of the clusterRole, and an error, if there is any.
func (c *FakeClusterRoles) Update(clusterRole *v1beta1.ClusterRole) (result *v1beta1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(clusterrolesResource, clusterRole), &v1beta1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ClusterRole), err
}

// Delete takes name of the clusterRole and deletes it. Returns an error if one occurs.
func (c *FakeClusterRoles) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(clusterrolesResource, name), &v1beta1.ClusterRole{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeClusterRoles) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(clusterrolesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.ClusterRoleList{})
	return err
}

// Patch applies the patch and returns the patched clusterRole.
func (c *FakeClusterRoles) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.ClusterRole, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(clusterrolesResource, name, data, subresources...), &v1beta1.ClusterRole{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ClusterRole), err
}
