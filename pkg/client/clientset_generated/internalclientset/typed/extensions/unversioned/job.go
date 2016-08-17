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

// JobsGetter has a method to return a JobInterface.
// A group's client should implement this interface.
type JobsGetter interface {
	Jobs(namespace string) JobInterface
}

// JobInterface has methods to work with Job resources.
type JobInterface interface {
	Create(*batch.Job) (*batch.Job, error)
	Update(*batch.Job) (*batch.Job, error)
	UpdateStatus(*batch.Job) (*batch.Job, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*batch.Job, error)
	List(opts api.ListOptions) (*batch.JobList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	JobExpansion
}

// jobs implements JobInterface
type jobs struct {
	client *ExtensionsClient
	ns     string
}

// newJobs returns a Jobs
func newJobs(c *ExtensionsClient, namespace string) *jobs {
	return &jobs{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a job and creates it.  Returns the server's representation of the job, and an error, if there is any.
func (c *jobs) Create(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("jobs").
		Body(job).
		Do().
		Into(result)
	return
}

// Update takes the representation of a job and updates it. Returns the server's representation of the job, and an error, if there is any.
func (c *jobs) Update(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("jobs").
		Name(job.Name).
		Body(job).
		Do().
		Into(result)
	return
}

func (c *jobs) UpdateStatus(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("jobs").
		Name(job.Name).
		SubResource("status").
		Body(job).
		Do().
		Into(result)
	return
}

// Delete takes name of the job and deletes it. Returns an error if one occurs.
func (c *jobs) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("jobs").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *jobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the job, and returns the corresponding job object, and an error if there is any.
func (c *jobs) Get(name string) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("jobs").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Jobs that match those selectors.
func (c *jobs) List(opts api.ListOptions) (result *batch.JobList, err error) {
	result = &batch.JobList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *jobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
