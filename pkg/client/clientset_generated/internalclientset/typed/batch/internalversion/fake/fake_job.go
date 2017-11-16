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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	batch "k8s.io/kubernetes/pkg/apis/batch"
)

// FakeJobs implements JobInterface
type FakeJobs struct {
	Fake *FakeBatch
	ns   string
}

var jobsResource = schema.GroupVersionResource{Group: "batch", Version: "", Resource: "jobs"}

var jobsKind = schema.GroupVersionKind{Group: "batch", Version: "", Kind: "Job"}

// Get takes name of the job, and returns the corresponding job object, and an error if there is any.
func (c *FakeJobs) Get(name string, options v1.GetOptions) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(jobsResource, c.ns, name), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

// List takes label and field selectors, and returns the list of Jobs that match those selectors.
func (c *FakeJobs) List(opts v1.ListOptions) (result *batch.JobList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(jobsResource, jobsKind, c.ns, opts), &batch.JobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakeJobs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(jobsResource, c.ns, opts))

}

// Create takes the representation of a job and creates it.  Returns the server's representation of the job, and an error, if there is any.
func (c *FakeJobs) Create(job *batch.Job) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(jobsResource, c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

// Update takes the representation of a job and updates it. Returns the server's representation of the job, and an error, if there is any.
func (c *FakeJobs) Update(job *batch.Job) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(jobsResource, c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeJobs) UpdateStatus(job *batch.Job) (*batch.Job, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(jobsResource, "status", c.ns, job), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}

// Delete takes name of the job and deletes it. Returns an error if one occurs.
func (c *FakeJobs) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(jobsResource, c.ns, name), &batch.Job{})

	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeJobs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(jobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &batch.JobList{})
	return err
}

// Patch applies the patch and returns the patched job.
func (c *FakeJobs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *batch.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(jobsResource, c.ns, name, data, subresources...), &batch.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.Job), err
}
