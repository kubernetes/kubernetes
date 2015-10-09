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
	"k8s.io/kubernetes/pkg/apis/extensions"
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

func (c *FakeIngress) Get(name string) (*extensions.Ingress, error) {
	obj, err := c.Fake.Invokes(NewGetAction("ingress", c.Namespace, name), &extensions.Ingress{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) List(label labels.Selector, fields fields.Selector) (*extensions.IngressList, error) {
	obj, err := c.Fake.Invokes(NewListAction("ingress", c.Namespace, label, nil), &extensions.IngressList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.IngressList), err
}

func (c *FakeIngress) Create(ingress *extensions.Ingress) (*extensions.Ingress, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("ingress", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) Update(ingress *extensions.Ingress) (*extensions.Ingress, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("ingress", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("ingress", c.Namespace, name), &extensions.Ingress{})
	return err
}

func (c *FakeIngress) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("ingress", c.Namespace, label, field, resourceVersion))
}

func (c *FakeIngress) UpdateStatus(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("ingress", "status", c.Namespace, ingress), ingress)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Ingress), err
}
