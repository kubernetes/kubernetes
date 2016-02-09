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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

type FakeClusters struct {
	Fake *Fake
}

func (c *FakeClusters) Get(name string) (*api.Cluster, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("clusters", name), &api.Cluster{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Cluster), err
}

func (c *FakeClusters) List(opts api.ListOptions) (*api.ClusterList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("clusters", opts), &api.ClusterList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ClusterList), err
}

func (c *FakeClusters) Create(cluster *api.Cluster) (*api.Cluster, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("clusters", cluster), cluster)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Cluster), err
}

func (c *FakeClusters) Update(cluster *api.Cluster) (*api.Cluster, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("clusters", cluster), cluster)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Cluster), err
}

func (c *FakeClusters) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("clusters", name), &api.Cluster{})
	return err
}

func (c *FakeClusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("clusters", opts))
}

func (c *FakeClusters) UpdateStatus(cluster *api.Cluster) (*api.Cluster, error) {
	action := CreateActionImpl{}
	action.Verb = "update"
	action.Resource = "clusters"
	action.Subresource = "status"
	action.Object = cluster

	obj, err := c.Fake.Invokes(action, cluster)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Cluster), err
}
