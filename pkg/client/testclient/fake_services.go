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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Fake implements ServiceInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeServices struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeServices) List(selector labels.Selector) (*api.ServiceList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-services"}, &api.ServiceList{})
	return obj.(*api.ServiceList), err
}

func (c *FakeServices) Get(name string) (*api.Service, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-service", Value: name}, &api.Service{})
	return obj.(*api.Service), err
}

func (c *FakeServices) Create(service *api.Service) (*api.Service, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-service", Value: service}, &api.Service{})
	return obj.(*api.Service), err
}

func (c *FakeServices) Update(service *api.Service) (*api.Service, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-service", Value: service}, &api.Service{})
	return obj.(*api.Service), err
}

func (c *FakeServices) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-service", Value: name}, &api.Service{})
	return err
}

func (c *FakeServices) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-services", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
