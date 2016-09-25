/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/watch"
)

// JobsNamespacer has methods to work with Job resources in a namespace
type JobsNamespacer interface {
	Jobs(namespace string) JobInterface
}

// JobInterface exposes methods to work on Job resources.
type JobInterface interface {
	List(opts api.ListOptions) (*batch.JobList, error)
	Get(name string) (*batch.Job, error)
	Create(job *batch.Job) (*batch.Job, error)
	Update(job *batch.Job) (*batch.Job, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(job *batch.Job) (*batch.Job, error)
}

// jobs implements JobsNamespacer interface
type jobs struct {
	r  *ExtensionsClient
	ns string
}

// newJobs returns a jobs
func newJobs(c *ExtensionsClient, namespace string) *jobs {
	return &jobs{c, namespace}
}

// Ensure statically that jobs implements JobInterface.
var _ JobInterface = &jobs{}

// List returns a list of jobs that match the label and field selectors.
func (c *jobs) List(opts api.ListOptions) (result *batch.JobList, err error) {
	result = &batch.JobList{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular job.
func (c *jobs) Get(name string) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").Name(name).Do().Into(result)
	return
}

// Create creates a new job.
func (c *jobs) Create(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Post().Namespace(c.ns).Resource("jobs").Body(job).Do().Into(result)
	return
}

// Update updates an existing job.
func (c *jobs) Update(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(job.Name).Body(job).Do().Into(result)
	return
}

// Delete deletes a job, returns error if one occurs.
func (c *jobs) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("jobs").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *jobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the job and the new status.  Returns the server's representation of the job, and an error, if it occurs.
func (c *jobs) UpdateStatus(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(job.Name).SubResource("status").Body(job).Do().Into(result)
	return
}

// jobsV1 implements JobsNamespacer interface using BatchClient internally
type jobsV1 struct {
	r  *BatchClient
	ns string
}

// newJobsV1 returns a jobsV1
func newJobsV1(c *BatchClient, namespace string) *jobsV1 {
	return &jobsV1{c, namespace}
}

// Ensure statically that jobsV1 implements JobInterface.
var _ JobInterface = &jobsV1{}

// List returns a list of jobs that match the label and field selectors.
func (c *jobsV1) List(opts api.ListOptions) (result *batch.JobList, err error) {
	result = &batch.JobList{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular job.
func (c *jobsV1) Get(name string) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").Name(name).Do().Into(result)
	return
}

// Create creates a new job.
func (c *jobsV1) Create(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Post().Namespace(c.ns).Resource("jobs").Body(job).Do().Into(result)
	return
}

// Update updates an existing job.
func (c *jobsV1) Update(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(job.Name).Body(job).Do().Into(result)
	return
}

// Delete deletes a job, returns error if one occurs.
func (c *jobsV1) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("jobs").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *jobsV1) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the job and the new status.  Returns the server's representation of the job, and an error, if it occurs.
func (c *jobsV1) UpdateStatus(job *batch.Job) (result *batch.Job, err error) {
	result = &batch.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(job.Name).SubResource("status").Body(job).Do().Into(result)
	return
}
