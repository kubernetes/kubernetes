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
	v1api "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeEndpoints implements EndpointInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeEndpoints struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeEndpoints) Create(endpoints *v1api.Endpoints) (*v1api.Endpoints, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-endpoints"}, &v1api.Endpoints{})
	return obj.(*v1api.Endpoints), err
}

func (c *FakeEndpoints) List(selector labels.Selector) (*v1api.EndpointsList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-endpoints"}, &v1api.EndpointsList{})
	return obj.(*v1api.EndpointsList), err
}

func (c *FakeEndpoints) Get(name string) (*v1api.Endpoints, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-endpoints", Value: name}, &v1api.Endpoints{})
	return obj.(*v1api.Endpoints), err
}

func (c *FakeEndpoints) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-endpoints", Value: name}, &v1api.Endpoints{})
	return err
}

func (c *FakeEndpoints) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-endpoints", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}

func (c *FakeEndpoints) Update(endpoints *v1api.Endpoints) (*v1api.Endpoints, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-endpoints", Value: endpoints.Name}, &v1api.Endpoints{})
	return obj.(*v1api.Endpoints), err
}
