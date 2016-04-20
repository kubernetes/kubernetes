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
	"k8s.io/kubernetes/pkg/watch"
)

// FakeWorkflows implements WorkflowInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeWorkflows struct {
	Fake      *FakeBatch
	Namespace string
}

func (c *FakeWorkflows) Get(name string) (*extensions.Workflow, error) {
	obj, err := c.Fake.Invokes(NewGetAction("workflows", c.Namespace, name), &extensions.Workflow{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Workflow), err
}

func (c *FakeWorkflows) List(opts api.ListOptions) (*extensions.WorkflowList, error) {
	obj, err := c.Fake.Invokes(NewListAction("workflows", c.Namespace, opts), &extensions.WorkflowList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.WorkflowList), err
}

func (c *FakeWorkflows) Create(workflow *extensions.Workflow) (*extensions.Workflow, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("workflows", c.Namespace, workflow), workflow)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Workflow), err
}

func (c *FakeWorkflows) Update(workflow *extensions.Workflow) (*extensions.Workflow, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("workflows", c.Namespace, workflow), workflow)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Workflow), err
}

func (c *FakeWorkflows) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("workflows", c.Namespace, name), &extensions.Workflow{})
	return err
}

func (c *FakeWorkflows) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("workflows", c.Namespace, opts))
}

func (c *FakeWorkflows) UpdateStatus(workflow *extensions.Workflow) (result *extensions.Workflow, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("workflows", "status", c.Namespace, workflow), workflow)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Workflow), err
}
