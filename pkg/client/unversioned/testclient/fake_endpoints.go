/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeEndpoints implements EndpointInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeEndpoints struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeEndpoints) Get(name string) (*api.Endpoints, error) {
	obj, err := c.Fake.Invokes(NewGetAction("endpoints", c.Namespace, name), &api.Endpoints{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) List(label labels.Selector) (*api.EndpointsList, error) {
	obj, err := c.Fake.Invokes(NewListAction("endpoints", c.Namespace, label, nil), &api.EndpointsList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.EndpointsList), err
}

func (c *FakeEndpoints) Create(endpoints *api.Endpoints) (*api.Endpoints, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("endpoints", c.Namespace, endpoints), endpoints)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) Update(endpoints *api.Endpoints) (*api.Endpoints, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("endpoints", c.Namespace, endpoints), endpoints)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("endpoints", c.Namespace, name), &api.Endpoints{})
	return err
}

func (c *FakeEndpoints) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("endpoints", c.Namespace, label, field, resourceVersion))
}
