/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeIngress implements IngressInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeIngress struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeIngress) Get(name string) (*experimental.Ingress, error) {
	obj, err := c.Fake.Invokes(NewGetAction("ingress", c.Namespace, name), &experimental.Ingress{})
	if obj == nil {
		return nil, err
	}

	return obj.(*experimental.Ingress), err
}

func (c *FakeIngress) List(label labels.Selector, fields fields.Selector) (*experimental.IngressList, error) {
	obj, err := c.Fake.Invokes(NewListAction("ingress", c.Namespace, label, nil), &experimental.IngressList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*experimental.IngressList), err
}

func (c *FakeIngress) Create(ingress *experimental.Ingress) (*experimental.Ingress, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("ingress", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*experimental.Ingress), err
}

func (c *FakeIngress) Update(ingress *experimental.Ingress) (*experimental.Ingress, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("ingress", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*experimental.Ingress), err
}

func (c *FakeIngress) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("ingress", c.Namespace, name), &experimental.Ingress{})
	return err
}

func (c *FakeIngress) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("ingress", c.Namespace, label, field, resourceVersion))
}

func (c *FakeIngress) UpdateStatus(ingress *experimental.Ingress) (result *experimental.Ingress, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("ingress", "status", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*experimental.Ingress), err
}
