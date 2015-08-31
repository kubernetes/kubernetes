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
	kClientLib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeDaemons implements DaemonInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeDaemons struct {
	Fake      *Fake
	Namespace string
}

// Ensure statically that FakeDaemons implements DaemonInterface.
var _ kClientLib.DaemonInterface = &FakeDaemons{}

func (c *FakeDaemons) Get(name string) (*expapi.Daemon, error) {
	obj, err := c.Fake.Invokes(NewGetAction("daemons", c.Namespace, name), &expapi.Daemon{})
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Daemon), err
}

func (c *FakeDaemons) List(label labels.Selector) (*expapi.DaemonList, error) {
	obj, err := c.Fake.Invokes(NewListAction("daemons", c.Namespace, label, nil), &expapi.DaemonList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.DaemonList), err
}

func (c *FakeDaemons) Create(daemon *expapi.Daemon) (*expapi.Daemon, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("daemons", c.Namespace, daemon), &expapi.Daemon{})
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Daemon), err
}

func (c *FakeDaemons) Update(daemon *expapi.Daemon) (*expapi.Daemon, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("daemons", c.Namespace, daemon), &expapi.Daemon{})
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Daemon), err
}

func (c *FakeDaemons) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("daemons", c.Namespace, name), &expapi.Daemon{})
	return err
}

func (c *FakeDaemons) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(NewWatchAction("daemons", c.Namespace, label, field, resourceVersion), nil)
	return c.Fake.Watch, nil
}
