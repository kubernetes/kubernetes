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

// Fake implements SecretInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeSecrets struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeSecrets) List(labels, fields labels.Selector) (*api.SecretList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-secrets"})
	return &c.Fake.SecretList, c.Fake.Err
}

func (c *FakeSecrets) Get(name string) (*api.Secret, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-secret", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.Secret).(*api.Secret), nil
}

func (c *FakeSecrets) Create(secret *api.Secret) (*api.Secret, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-secret", Value: secret})
	return &api.Secret{}, nil
}

func (c *FakeSecrets) Update(secret *api.Secret) (*api.Secret, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-secret", Value: secret})
	return &api.Secret{}, nil
}

func (c *FakeSecrets) Delete(secret string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-secret", Value: secret})
	return nil
}

func (c *FakeSecrets) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-secrets", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
