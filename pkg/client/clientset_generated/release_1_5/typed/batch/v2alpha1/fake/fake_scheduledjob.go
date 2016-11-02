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
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeScheduledJobs implements ScheduledJobInterface
type FakeScheduledJobs struct {
	Fake *FakeBatchV2alpha1
	ns   string
}

var scheduledjobsResource = unversioned.GroupVersionResource{Group: "batch", Version: "v2alpha1", Resource: "scheduledjobs"}

func (c *FakeScheduledJobs) Create(scheduledJob *v2alpha1.ScheduledJob) (result *v2alpha1.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(scheduledjobsResource, c.ns, scheduledJob), &v2alpha1.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.ScheduledJob), err
}

func (c *FakeScheduledJobs) Update(scheduledJob *v2alpha1.ScheduledJob) (result *v2alpha1.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(scheduledjobsResource, c.ns, scheduledJob), &v2alpha1.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.ScheduledJob), err
}

func (c *FakeScheduledJobs) UpdateStatus(scheduledJob *v2alpha1.ScheduledJob) (*v2alpha1.ScheduledJob, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(scheduledjobsResource, "status", c.ns, scheduledJob), &v2alpha1.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.ScheduledJob), err
}

func (c *FakeScheduledJobs) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(scheduledjobsResource, c.ns, name), &v2alpha1.ScheduledJob{})

	return err
}

func (c *FakeScheduledJobs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(scheduledjobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v2alpha1.ScheduledJobList{})
	return err
}

func (c *FakeScheduledJobs) Get(name string) (result *v2alpha1.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(scheduledjobsResource, c.ns, name), &v2alpha1.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.ScheduledJob), err
}

func (c *FakeScheduledJobs) List(opts v1.ListOptions) (result *v2alpha1.ScheduledJobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(scheduledjobsResource, c.ns, opts), &v2alpha1.ScheduledJobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v2alpha1.ScheduledJobList{}
	for _, item := range obj.(*v2alpha1.ScheduledJobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested scheduledJobs.
func (c *FakeScheduledJobs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(scheduledjobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched scheduledJob.
func (c *FakeScheduledJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v2alpha1.ScheduledJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(scheduledjobsResource, c.ns, name, data, subresources...), &v2alpha1.ScheduledJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.ScheduledJob), err
}
