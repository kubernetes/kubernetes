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
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeStatefulSets implements StatefulSetsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeStatefulSets struct {
	Fake      *FakeApps
	Namespace string
}

func (c *FakeStatefulSets) Get(name string) (*apps.StatefulSet, error) {
	obj, err := c.Fake.Invokes(NewGetAction("statefulsets", c.Namespace, name), &apps.StatefulSet{})
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) List(opts api.ListOptions) (*apps.StatefulSetList, error) {
	obj, err := c.Fake.Invokes(NewListAction("statefulsets", c.Namespace, opts), &apps.StatefulSetList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSetList), err
}

func (c *FakeStatefulSets) Create(rs *apps.StatefulSet) (*apps.StatefulSet, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("statefulsets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) Update(rs *apps.StatefulSet) (*apps.StatefulSet, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("statefulsets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("statefulsets", c.Namespace, name), &apps.StatefulSet{})
	return err
}

func (c *FakeStatefulSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("statefulsets", c.Namespace, opts))
}

func (c *FakeStatefulSets) UpdateStatus(rs *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("statefulsets", "status", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.StatefulSet), err
}
