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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeNamespaces implements NamespacesInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeNamespaces struct {
	Fake *Fake
}

func (c *FakeNamespaces) List(labels labels.Selector, field fields.Selector) (*api.NamespaceList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-namespaces"})
	return api.Scheme.CopyOrDie(&c.Fake.NamespacesList).(*api.NamespaceList), nil
}

func (c *FakeNamespaces) Get(name string) (*api.Namespace, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-namespace", Value: name})
	return &api.Namespace{ObjectMeta: api.ObjectMeta{Name: name}}, nil
}

func (c *FakeNamespaces) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-namespace", Value: name})
	return nil
}

func (c *FakeNamespaces) Create(namespace *api.Namespace) (*api.Namespace, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-namespace"})
	return &api.Namespace{}, c.Fake.Err
}

func (c *FakeNamespaces) Update(namespace *api.Namespace) (*api.Namespace, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-namespace", Value: namespace.Name})
	return &api.Namespace{}, nil
}

func (c *FakeNamespaces) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-namespaces", Value: resourceVersion})
	return c.Fake.Watch, nil
}

func (c *FakeNamespaces) Finalize(namespace *api.Namespace) (*api.Namespace, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "finalize-namespace", Value: namespace.Name})
	return &api.Namespace{}, nil
}

func (c *FakeNamespaces) Status(namespace *api.Namespace) (*api.Namespace, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "status-namespace", Value: namespace.Name})
	return &api.Namespace{}, nil
}
