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

// FakeServiceAccounts implements ServiceAccountsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeServiceAccounts struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeServiceAccounts) Get(name string) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(NewGetAction("serviceaccounts", c.Namespace, name), &api.ServiceAccount{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) List(label labels.Selector, field fields.Selector) (*api.ServiceAccountList, error) {
	obj, err := c.Fake.Invokes(NewListAction("serviceaccounts", c.Namespace, label, field), &api.ServiceAccountList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ServiceAccountList), err
}

func (c *FakeServiceAccounts) Create(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("serviceaccounts", c.Namespace, serviceAccount), serviceAccount)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Update(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("serviceaccounts", c.Namespace, serviceAccount), serviceAccount)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("serviceaccounts", c.Namespace, name), &api.ServiceAccount{})
	return err
}

func (c *FakeServiceAccounts) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(NewWatchAction("serviceaccounts", c.Namespace, label, field, resourceVersion), nil)
	return c.Fake.Watch, c.Fake.Err()
}
