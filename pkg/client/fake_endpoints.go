/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeEndpoints implements EndpointInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeEndpoints struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeEndpoints) Create(endpoints *api.Endpoints) (*api.Endpoints, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-endpoints"})
	return &api.Endpoints{}, nil
}

func (c *FakeEndpoints) List(selector labels.Selector) (*api.EndpointsList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-endpoints"})
	return api.Scheme.CopyOrDie(&c.Fake.EndpointsList).(*api.EndpointsList), c.Fake.Err
}

func (c *FakeEndpoints) Get(name string) (*api.Endpoints, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-endpoints"})
	return &api.Endpoints{}, nil
}

func (c *FakeEndpoints) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-endpoints", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}

func (c *FakeEndpoints) Update(endpoints *api.Endpoints) (*api.Endpoints, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-endpoints", Value: endpoints.Name})
	return &api.Endpoints{}, nil
}
