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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	v1 "k8s.io/kubernetes/pkg/apis/batch/v1"
)

// FakeJobs implements JobInterface
type FakeJobs struct {
	Fake *FakeBatchV1
	ns   string
}

var jobsResource = schema.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"}

func (c *FakeJobs) Create(job *v1.Job) (result *v1.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(jobsResource, c.ns, job), &v1.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Job), err
}

func (c *FakeJobs) Update(job *v1.Job) (result *v1.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(jobsResource, c.ns, job), &v1.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Job), err
}

func (c *FakeJobs) UpdateStatus(job *v1.Job) (*v1.Job, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(jobsResource, "status", c.ns, job), &v1.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Job), err
}

func (c *FakeJobs) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(jobsResource, c.ns, name), &v1.Job{})

	return err
}

func (c *FakeJobs) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(jobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.JobList{})
	return err
}

func (c *FakeJobs) Get(name string, options meta_v1.GetOptions) (result *v1.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(jobsResource, c.ns, name), &v1.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Job), err
}

func (c *FakeJobs) List(opts meta_v1.ListOptions) (result *v1.JobList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(jobsResource, c.ns, opts), &v1.JobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.JobList{}
	for _, item := range obj.(*v1.JobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *FakeJobs) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(jobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched job.
func (c *FakeJobs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.Job, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(jobsResource, c.ns, name, data, subresources...), &v1.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Job), err
}
