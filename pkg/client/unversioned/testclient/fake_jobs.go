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

// FakeJobs implements JobInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeJobs struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeJobs) Get(name string) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewGetAction("jobs", c.Namespace, name), &extensions.Job{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobs) List(opts api.ListOptions) (*extensions.JobList, error) {
	obj, err := c.Fake.Invokes(NewListAction("jobs", c.Namespace, opts), &extensions.JobList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.JobList), err
}

func (c *FakeJobs) Create(job *extensions.Job) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("jobs", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobs) Update(job *extensions.Job) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("jobs", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("jobs", c.Namespace, name), &extensions.Job{})
	return err
}

func (c *FakeJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("jobs", c.Namespace, opts))
}

func (c *FakeJobs) UpdateStatus(job *extensions.Job) (result *extensions.Job, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("jobs", "status", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

// FakeJobs implements JobInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
// This is a test implementation of JobsV1
// TODO(piosz): get back to one client implementation once HPA will be graduated to GA completely
type FakeJobsV1 struct {
	Fake      *FakeBatch
	Namespace string
}

func (c *FakeJobsV1) Get(name string) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewGetAction("jobs", c.Namespace, name), &extensions.Job{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobsV1) List(opts api.ListOptions) (*extensions.JobList, error) {
	obj, err := c.Fake.Invokes(NewListAction("jobs", c.Namespace, opts), &extensions.JobList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.JobList), err
}

func (c *FakeJobsV1) Create(job *extensions.Job) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("jobs", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobsV1) Update(job *extensions.Job) (*extensions.Job, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("jobs", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}

func (c *FakeJobsV1) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("jobs", c.Namespace, name), &extensions.Job{})
	return err
}

func (c *FakeJobsV1) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("jobs", c.Namespace, opts))
}

func (c *FakeJobsV1) UpdateStatus(job *extensions.Job) (result *extensions.Job, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("jobs", "status", c.Namespace, job), job)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Job), err
}
