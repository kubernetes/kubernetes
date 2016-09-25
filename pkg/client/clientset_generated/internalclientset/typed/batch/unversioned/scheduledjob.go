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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	batch "k8s.io/kubernetes/pkg/apis/batch"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ScheduledJobsGetter has a method to return a ScheduledJobInterface.
// A group's client should implement this interface.
type ScheduledJobsGetter interface {
	ScheduledJobs(namespace string) ScheduledJobInterface
}

// ScheduledJobInterface has methods to work with ScheduledJob resources.
type ScheduledJobInterface interface {
	Create(*batch.ScheduledJob) (*batch.ScheduledJob, error)
	Update(*batch.ScheduledJob) (*batch.ScheduledJob, error)
	UpdateStatus(*batch.ScheduledJob) (*batch.ScheduledJob, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*batch.ScheduledJob, error)
	List(opts api.ListOptions) (*batch.ScheduledJobList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *batch.ScheduledJob, err error)
	ScheduledJobExpansion
}

// scheduledJobs implements ScheduledJobInterface
type scheduledJobs struct {
	client *BatchClient
	ns     string
}

// newScheduledJobs returns a ScheduledJobs
func newScheduledJobs(c *BatchClient, namespace string) *scheduledJobs {
	return &scheduledJobs{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a scheduledJob and creates it.  Returns the server's representation of the scheduledJob, and an error, if there is any.
func (c *scheduledJobs) Create(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("scheduledjobs").
		Body(scheduledJob).
		Do().
		Into(result)
	return
}

// Update takes the representation of a scheduledJob and updates it. Returns the server's representation of the scheduledJob, and an error, if there is any.
func (c *scheduledJobs) Update(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("scheduledjobs").
		Name(scheduledJob.Name).
		Body(scheduledJob).
		Do().
		Into(result)
	return
}

func (c *scheduledJobs) UpdateStatus(scheduledJob *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("scheduledjobs").
		Name(scheduledJob.Name).
		SubResource("status").
		Body(scheduledJob).
		Do().
		Into(result)
	return
}

// Delete takes name of the scheduledJob and deletes it. Returns an error if one occurs.
func (c *scheduledJobs) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("scheduledjobs").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *scheduledJobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("scheduledjobs").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the scheduledJob, and returns the corresponding scheduledJob object, and an error if there is any.
func (c *scheduledJobs) Get(name string) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("scheduledjobs").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ScheduledJobs that match those selectors.
func (c *scheduledJobs) List(opts api.ListOptions) (result *batch.ScheduledJobList, err error) {
	result = &batch.ScheduledJobList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("scheduledjobs").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested scheduledJobs.
func (c *scheduledJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("scheduledjobs").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched scheduledJob.
func (c *scheduledJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("scheduledjobs").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
