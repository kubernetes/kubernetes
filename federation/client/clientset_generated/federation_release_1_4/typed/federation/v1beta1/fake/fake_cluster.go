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
	v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeClusters implements ClusterInterface
type FakeClusters struct {
	Fake *FakeFederation
}

var clustersResource = unversioned.GroupVersionResource{Group: "federation", Version: "v1beta1", Resource: "clusters"}

func (c *FakeClusters) Create(cluster *v1beta1.Cluster) (result *v1beta1.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(clustersResource, cluster), &v1beta1.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Cluster), err
}

func (c *FakeClusters) Update(cluster *v1beta1.Cluster) (result *v1beta1.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(clustersResource, cluster), &v1beta1.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Cluster), err
}

func (c *FakeClusters) UpdateStatus(cluster *v1beta1.Cluster) (*v1beta1.Cluster, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(clustersResource, "status", cluster), &v1beta1.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Cluster), err
}

func (c *FakeClusters) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(clustersResource, name), &v1beta1.Cluster{})
	return err
}

func (c *FakeClusters) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(clustersResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.ClusterList{})
	return err
}

func (c *FakeClusters) Get(name string) (result *v1beta1.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(clustersResource, name), &v1beta1.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Cluster), err
}

func (c *FakeClusters) List(opts api.ListOptions) (result *v1beta1.ClusterList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(clustersResource, opts), &v1beta1.ClusterList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.ClusterList{}
	for _, item := range obj.(*v1beta1.ClusterList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusters.
func (c *FakeClusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(clustersResource, opts))
}

// Patch applies the patch and returns the patched cluster.
func (c *FakeClusters) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(clustersResource, name, data, subresources...), &v1beta1.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Cluster), err
}
