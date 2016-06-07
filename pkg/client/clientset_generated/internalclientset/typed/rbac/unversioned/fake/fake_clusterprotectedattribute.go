/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// FakeClusterProtectedAttributes implements ClusterProtectedAttributeInterface
type FakeClusterProtectedAttributes struct {
	Fake *FakeRbac
}

var clusterprotectedattributesResource = unversioned.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "", Resource: "clusterprotectedattributes"}

func (c *FakeClusterProtectedAttributes) Create(clusterProtectedAttribute *rbac.ClusterProtectedAttribute) (result *rbac.ClusterProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(clusterprotectedattributesResource, clusterProtectedAttribute), &rbac.ClusterProtectedAttribute{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttribute), err
}

func (c *FakeClusterProtectedAttributes) Update(clusterProtectedAttribute *rbac.ClusterProtectedAttribute) (result *rbac.ClusterProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(clusterprotectedattributesResource, clusterProtectedAttribute), &rbac.ClusterProtectedAttribute{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttribute), err
}

func (c *FakeClusterProtectedAttributes) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(clusterprotectedattributesResource, name), &rbac.ClusterProtectedAttribute{})
	return err
}

func (c *FakeClusterProtectedAttributes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(clusterprotectedattributesResource, listOptions)

	_, err := c.Fake.Invokes(action, &rbac.ClusterProtectedAttributeList{})
	return err
}

func (c *FakeClusterProtectedAttributes) Get(name string) (result *rbac.ClusterProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(clusterprotectedattributesResource, name), &rbac.ClusterProtectedAttribute{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttribute), err
}

func (c *FakeClusterProtectedAttributes) List(opts api.ListOptions) (result *rbac.ClusterProtectedAttributeList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(clusterprotectedattributesResource, opts), &rbac.ClusterProtectedAttributeList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &rbac.ClusterProtectedAttributeList{}
	for _, item := range obj.(*rbac.ClusterProtectedAttributeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusterProtectedAttributes.
func (c *FakeClusterProtectedAttributes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(clusterprotectedattributesResource, opts))
}

// Patch applies the patch and returns the patched clusterProtectedAttribute.
func (c *FakeClusterProtectedAttributes) Patch(name string, pt api.PatchType, data []byte) (result *rbac.ClusterProtectedAttribute, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchAction(clusterprotectedattributesResource, name, data), &rbac.ClusterProtectedAttribute{})
	if obj == nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttribute), err
}
