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
	"k8s.io/kubernetes/pkg/watch"
)

const (
	secretResourceName string = "secrets"
)

// Fake implements SecretInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeSecrets struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeSecrets) Get(name string) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(NewGetAction(secretResourceName, c.Namespace, name), &api.Secret{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Secret), err
}

func (c *FakeSecrets) List(opts api.ListOptions) (*api.SecretList, error) {
	obj, err := c.Fake.Invokes(NewListAction(secretResourceName, c.Namespace, opts), &api.SecretList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.SecretList), err
}

func (c *FakeSecrets) Create(secret *api.Secret) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(NewCreateAction(secretResourceName, c.Namespace, secret), secret)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Update(secret *api.Secret) (*api.Secret, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction(secretResourceName, c.Namespace, secret), secret)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction(secretResourceName, c.Namespace, name), &api.Secret{})
	return err
}

func (c *FakeSecrets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction(secretResourceName, c.Namespace, opts))
}
