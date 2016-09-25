/*
Copyright 2014 The Kubernetes Authors.

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

// FakeLimitRanges implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeLimitRanges struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeLimitRanges) Get(name string) (*api.LimitRange, error) {
	obj, err := c.Fake.Invokes(NewGetAction("limitranges", c.Namespace, name), &api.LimitRange{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) List(opts api.ListOptions) (*api.LimitRangeList, error) {
	obj, err := c.Fake.Invokes(NewListAction("limitranges", c.Namespace, opts), &api.LimitRangeList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.LimitRangeList), err
}

func (c *FakeLimitRanges) Create(limitRange *api.LimitRange) (*api.LimitRange, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("limitranges", c.Namespace, limitRange), limitRange)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) Update(limitRange *api.LimitRange) (*api.LimitRange, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("limitranges", c.Namespace, limitRange), limitRange)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("limitranges", c.Namespace, name), &api.LimitRange{})
	return err
}

func (c *FakeLimitRanges) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("limitranges", c.Namespace, opts))
}
