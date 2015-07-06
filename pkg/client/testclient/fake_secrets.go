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

// Fake implements SecretInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeSecrets struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeSecrets) List(labels labels.Selector, field fields.Selector) (*api.SecretList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-secrets"}, &api.SecretList{})
	return obj.(*api.SecretList), err
}

func (c *FakeSecrets) Get(name string) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-secret", Value: name}, &api.Secret{})
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Create(secret *api.Secret) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-secret", Value: secret}, &api.Secret{})
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Update(secret *api.Secret) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-secret", Value: secret}, &api.Secret{})
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-secret", Value: name}, &api.Secret{})
	return err
}

func (c *FakeSecrets) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(FakeAction{Action: "watch-secrets", Value: resourceVersion}, nil)
	return c.Fake.Watch, c.Fake.Err()
}
