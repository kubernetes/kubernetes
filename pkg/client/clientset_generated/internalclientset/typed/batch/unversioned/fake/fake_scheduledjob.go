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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	batch "k8s.io/kubernetes/pkg/apis/batch"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeScheduledJobs implements ScheduledJobInterface
type FakeScheduledJobs struct {
	Fake *FakeBatch
	ns   string
}

var scheduledjobsResource = unversioned.GroupVersionResource{Group: "batch", Version: "", Resource: "scheduledjobs"}

func (c *FakeScheduledJobs) Create(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(scheduledjobsResource, c.ns, scheduledJob), &batch.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) Update(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(scheduledjobsResource, c.ns, scheduledJob), &batch.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) UpdateStatus(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(scheduledjobsResource, "status", c.ns, scheduledJob), &batch.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(scheduledjobsResource, c.ns, name), &batch.ScheduledJob{})

	return err
}

func (c *FakeScheduledJobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(scheduledjobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &batch.ScheduledJobList{})
	return err
}

func (c *FakeScheduledJobs) Get(name string) (result *batch.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(scheduledjobsResource, c.ns, name), &batch.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.ScheduledJob), err
}

func (c *FakeScheduledJobs) List(opts api.ListOptions) (result *batch.ScheduledJobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(scheduledjobsResource, c.ns, opts), &batch.ScheduledJobList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &batch.ScheduledJobList{}
	for _, item := range obj.(*batch.ScheduledJobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested scheduledJobs.
func (c *FakeScheduledJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(scheduledjobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched scheduledJob.
func (c *FakeScheduledJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *batch.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(scheduledjobsResource, c.ns, name, data, subresources...), &batch.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.ScheduledJob), err
}
