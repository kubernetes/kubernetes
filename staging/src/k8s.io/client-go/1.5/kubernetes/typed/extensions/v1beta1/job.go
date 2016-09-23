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

package v1beta1

import (
	api "k8s.io/client-go/1.5/pkg/api"
	v1beta1 "k8s.io/client-go/1.5/pkg/apis/extensions/v1beta1"
	watch "k8s.io/client-go/1.5/pkg/watch"
)

// JobsGetter has a method to return a JobInterface.
// A group's client should implement this interface.
type JobsGetter interface {
	Jobs(namespace string) JobInterface
}

// JobInterface has methods to work with Job resources.
type JobInterface interface {
	Create(*v1beta1.Job) (*v1beta1.Job, error)
	Update(*v1beta1.Job) (*v1beta1.Job, error)
	UpdateStatus(*v1beta1.Job) (*v1beta1.Job, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1beta1.Job, error)
	List(opts api.ListOptions) (*v1beta1.JobList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.Job, err error)
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
func (c *jobs) Create(job *v1beta1.Job) (result *v1beta1.Job, err error) {
	result = &v1beta1.Job{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("jobs").
		Body(job).
		Do().
		Into(result)
	return
}

// Update takes the representation of a job and updates it. Returns the server's representation of the job, and an error, if there is any.
func (c *jobs) Update(job *v1beta1.Job) (result *v1beta1.Job, err error) {
	result = &v1beta1.Job{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("jobs").
		Name(job.Name).
		Body(job).
		Do().
		Into(result)
	return
}

func (c *jobs) UpdateStatus(job *v1beta1.Job) (result *v1beta1.Job, err error) {
	result = &v1beta1.Job{}
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
func (c *jobs) Get(name string) (result *v1beta1.Job, err error) {
	result = &v1beta1.Job{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("jobs").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Jobs that match those selectors.
func (c *jobs) List(opts api.ListOptions) (result *v1beta1.JobList, err error) {
	result = &v1beta1.JobList{}
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

// Patch applies the patch and returns the patched job.
func (c *jobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.Job, err error) {
	result = &v1beta1.Job{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("jobs").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
