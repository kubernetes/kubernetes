/*
Copyright 2017 The Kubernetes Authors.

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

// FakeJobs implements JobInterface
type FakeJobs struct {
	Fake *FakeBatch
	ns   string
}

var jobsResource = unversioned.GroupVersionResource{Group: "batch", Version: "", Resource: "jobs"}

func (c *FakeJobs) Create(job *batch.Job) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(jobsResource, c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

func (c *FakeJobs) Update(job *batch.Job) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(jobsResource, c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

func (c *FakeJobs) UpdateStatus(job *batch.Job) (*batch.Job, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(jobsResource, "status", c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

func (c *FakeJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(jobsResource, c.ns, name), &batch.Job{})

	return err
}

func (c *FakeJobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(jobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &batch.JobList{})
	return err
}

func (c *FakeJobs) Get(name string) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(jobsResource, c.ns, name), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

func (c *FakeJobs) List(opts api.ListOptions) (result *batch.JobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(jobsResource, c.ns, opts), &batch.JobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &batch.JobList{}
	for _, item := range obj.(*batch.JobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *FakeJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(jobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched job.
func (c *FakeJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(jobsResource, c.ns, name, data, subresources...), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}
