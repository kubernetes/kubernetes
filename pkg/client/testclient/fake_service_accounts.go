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

// FakeServiceAccounts implements ServiceAccountsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeServiceAccounts struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeServiceAccounts) List(labels labels.Selector, field fields.Selector) (*api.ServiceAccountList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-serviceaccounts"}, &api.ServiceAccountList{})
	return obj.(*api.ServiceAccountList), err
}

func (c *FakeServiceAccounts) Get(name string) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-serviceaccount", Value: name}, &api.ServiceAccount{})
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Create(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-serviceaccount", Value: serviceAccount}, &api.ServiceAccount{})
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Update(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-serviceaccount", Value: serviceAccount}, &api.ServiceAccount{})
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-serviceaccount", Value: name}, &api.ServiceAccount{})
	return err
}

func (c *FakeServiceAccounts) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(FakeAction{Action: "watch-serviceAccounts", Value: resourceVersion}, nil)
	return c.Fake.Watch, c.Fake.Err()
}
