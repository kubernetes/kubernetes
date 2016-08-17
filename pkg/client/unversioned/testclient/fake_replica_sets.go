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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeReplicaSets implements ReplicaSetsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeReplicaSets struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeReplicaSets) Get(name string) (*extensions.ReplicaSet, error) {
	obj, err := c.Fake.Invokes(NewGetAction("replicasets", c.Namespace, name), &extensions.ReplicaSet{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ReplicaSet), err
}

func (c *FakeReplicaSets) List(opts api.ListOptions) (*extensions.ReplicaSetList, error) {
	obj, err := c.Fake.Invokes(NewListAction("replicasets", c.Namespace, opts), &extensions.ReplicaSetList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ReplicaSetList), err
}

func (c *FakeReplicaSets) Create(rs *extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("replicasets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ReplicaSet), err
}

func (c *FakeReplicaSets) Update(rs *extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("replicasets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ReplicaSet), err
}

func (c *FakeReplicaSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("replicasets", c.Namespace, name), &extensions.ReplicaSet{})
	return err
}

func (c *FakeReplicaSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("replicasets", c.Namespace, opts))
}

func (c *FakeReplicaSets) UpdateStatus(rs *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("replicasets", "status", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.ReplicaSet), err
}
