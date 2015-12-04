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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kclientlib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeDedicatedMachines implements DedicatedMachineInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeDedicatedMachines struct {
	Fake      *FakeExperimental
	Namespace string
}

// Ensure statically that FakeDedicatedMachines implements DedicatedMachineInterface.
var _ kclientlib.DedicatedMachineInterface = &FakeDedicatedMachines{}

func (c *FakeDedicatedMachines) Get(name string) (*extensions.DedicatedMachine, error) {
	obj, err := c.Fake.Invokes(NewGetAction("dedicatedmachines", c.Namespace, name), &extensions.DedicatedMachine{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.DedicatedMachine), err
}

func (c *FakeDedicatedMachines) List(label labels.Selector, field fields.Selector, opts unversioned.ListOptions) (*extensions.DedicatedMachineList, error) {
	obj, err := c.Fake.Invokes(NewListAction("dedicatedmachines", c.Namespace, label, field), &extensions.DedicatedMachineList{})
	if obj == nil {
		return nil, err
	}
	list := &extensions.DedicatedMachineList{}
	for _, dedicatedMachine := range obj.(*extensions.DedicatedMachineList).Items {
		if label.Matches(labels.Set(dedicatedMachine.Labels)) {
			list.Items = append(list.Items, dedicatedMachine)
		}
	}
	return list, err
}

func (c *FakeDedicatedMachines) Create(dedicatedMachine *extensions.DedicatedMachine) (*extensions.DedicatedMachine, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("dedicatedmachines", c.Namespace, dedicatedMachine), dedicatedMachine)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.DedicatedMachine), err
}

func (c *FakeDedicatedMachines) Update(dedicatedMachine *extensions.DedicatedMachine) (*extensions.DedicatedMachine, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("dedicatedmachines", c.Namespace, dedicatedMachine), dedicatedMachine)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.DedicatedMachine), err
}

func (c *FakeDedicatedMachines) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("dedicatedmachines", c.Namespace, name), &extensions.DedicatedMachine{})
	return err
}

func (c *FakeDedicatedMachines) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("dedicatedmachines", c.Namespace, opts))
}
