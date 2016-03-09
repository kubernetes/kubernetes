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
	controlplane "k8s.io/kubernetes/pkg/apis/controlplane"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeClusters implements ClusterInterface
type FakeClusters struct {
	Fake *FakeControlplane
}

func (c *FakeClusters) Create(cluster *controlplane.Cluster) (result *controlplane.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction("clusters", cluster), &controlplane.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*controlplane.Cluster), err
}

func (c *FakeClusters) Update(cluster *controlplane.Cluster) (result *controlplane.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction("clusters", cluster), &controlplane.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*controlplane.Cluster), err
}

func (c *FakeClusters) UpdateStatus(cluster *controlplane.Cluster) (*controlplane.Cluster, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction("clusters", "status", cluster), &controlplane.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*controlplane.Cluster), err
}

func (c *FakeClusters) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction("clusters", name), &controlplane.Cluster{})
	return err
}

func (c *FakeClusters) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction("clusters", listOptions)

	_, err := c.Fake.Invokes(action, &controlplane.ClusterList{})
	return err
}

func (c *FakeClusters) Get(name string) (result *controlplane.Cluster, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction("clusters", name), &controlplane.Cluster{})
	if obj == nil {
		return nil, err
	}
	return obj.(*controlplane.Cluster), err
}

func (c *FakeClusters) List(opts api.ListOptions) (result *controlplane.ClusterList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction("clusters", opts), &controlplane.ClusterList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &controlplane.ClusterList{}
	for _, item := range obj.(*controlplane.ClusterList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested clusters.
func (c *FakeClusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction("clusters", opts))
}
