/*
Copyright 2015 The Kubernetes Authors.

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
	kclientlib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeDaemonSet implements DaemonInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeDaemonSets struct {
	Fake      *FakeExperimental
	Namespace string
}

// Ensure statically that FakeDaemonSets implements DaemonInterface.
var _ kclientlib.DaemonSetInterface = &FakeDaemonSets{}

func (c *FakeDaemonSets) Get(name string) (*extensions.DaemonSet, error) {
	obj, err := c.Fake.Invokes(NewGetAction("daemonsets", c.Namespace, name), &extensions.DaemonSet{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.DaemonSet), err
}

func (c *FakeDaemonSets) List(opts api.ListOptions) (*extensions.DaemonSetList, error) {
	obj, err := c.Fake.Invokes(NewListAction("daemonsets", c.Namespace, opts), &extensions.DaemonSetList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.DaemonSetList), err
}

func (c *FakeDaemonSets) Create(daemon *extensions.DaemonSet) (*extensions.DaemonSet, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("daemonsets", c.Namespace, daemon), &extensions.DaemonSet{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.DaemonSet), err
}

func (c *FakeDaemonSets) Update(daemon *extensions.DaemonSet) (*extensions.DaemonSet, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("daemonsets", c.Namespace, daemon), &extensions.DaemonSet{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.DaemonSet), err
}

func (c *FakeDaemonSets) UpdateStatus(daemon *extensions.DaemonSet) (*extensions.DaemonSet, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("daemonsets", "status", c.Namespace, daemon), &extensions.DaemonSet{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.DaemonSet), err
}

func (c *FakeDaemonSets) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("daemonsets", c.Namespace, name), &extensions.DaemonSet{})
	return err
}

func (c *FakeDaemonSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("daemonsets", c.Namespace, opts))
}
