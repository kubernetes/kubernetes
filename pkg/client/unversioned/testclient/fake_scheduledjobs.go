/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeScheduledJobs implements ScheduledJobInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeScheduledJobs struct {
	Fake      *FakeBatch
	Namespace string
}

func (c *FakeScheduledJobs) Get(name string) (*batch.ScheduledJob, error) {
	obj, err := c.Fake.Invokes(NewGetAction("scheduledjobs", c.Namespace, name), &batch.ScheduledJob{})
	if obj == nil {
		return nil, err
	}

	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) List(opts api.ListOptions) (*batch.ScheduledJobList, error) {
	obj, err := c.Fake.Invokes(NewListAction("scheduledjobs", c.Namespace, opts), &batch.ScheduledJobList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*batch.ScheduledJobList), err
}

func (c *FakeScheduledJobs) Create(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("scheduledjobs", c.Namespace, scheduledJob), scheduledJob)
	if obj == nil {
		return nil, err
	}

	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) Update(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("scheduledjobs", c.Namespace, scheduledJob), scheduledJob)
	if obj == nil {
		return nil, err
	}

	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("scheduledjobs", c.Namespace, name), &batch.ScheduledJob{})
	return err
}

func (c *FakeScheduledJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("scheduledjobs", c.Namespace, opts))
}

func (c *FakeScheduledJobs) UpdateStatus(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("scheduledjobs", "status", c.Namespace, scheduledJob), scheduledJob)
	if obj == nil {
		return nil, err
	}

	return obj.(*batch.ScheduledJob), err
}
