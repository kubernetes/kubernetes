/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// ScheduledJobsNamespacer has methods to work with ScheduledJob resources in a namespace
type ScheduledJobsNamespacer interface {
	ScheduledJobs(namespace string) ScheduledJobInterface
}

// ScheduledJobInterface exposes methods to work on ScheduledJob resources.
type ScheduledJobInterface interface {
	List(opts api.ListOptions) (*batch.ScheduledJobList, error)
	Get(name string) (*batch.ScheduledJob, error)
	Create(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error)
	Update(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(scheduledJob *batch.ScheduledJob) (*batch.ScheduledJob, error)
}

// scheduledJobs implements ScheduledJobsNamespacer interface
type scheduledJobs struct {
	r  *BatchClient
	ns string
}

// newScheduledJobs returns a scheduledJobs
func newScheduledJobs(c *BatchClient, namespace string) *scheduledJobs {
	return &scheduledJobs{c, namespace}
}

// Ensure statically that scheduledJobs implements ScheduledJobInterface.
var _ ScheduledJobInterface = &scheduledJobs{}

// List returns a list of scheduled jobs that match the label and field selectors.
func (c *scheduledJobs) List(opts api.ListOptions) (result *batch.ScheduledJobList, err error) {
	result = &batch.ScheduledJobList{}
	err = c.r.Get().Namespace(c.ns).Resource("scheduledJobs").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular scheduled job.
func (c *scheduledJobs) Get(name string) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.r.Get().Namespace(c.ns).Resource("scheduledJobs").Name(name).Do().Into(result)
	return
}

// Create creates a new scheduled job.
func (c *scheduledJobs) Create(job *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.r.Post().Namespace(c.ns).Resource("scheduledJobs").Body(job).Do().Into(result)
	return
}

// Update updates an existing scheduled job.
func (c *scheduledJobs) Update(job *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.r.Put().Namespace(c.ns).Resource("scheduledJobs").Name(job.Name).Body(job).Do().Into(result)
	return
}

// Delete deletes a scheduled job, returns error if one occurs.
func (c *scheduledJobs) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("scheduledJobs").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested scheduled jobs.
func (c *scheduledJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("scheduledJobs").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the scheduled job and the new status.  Returns the server's representation of the scheduled job, and an error, if it occurs.
func (c *scheduledJobs) UpdateStatus(job *batch.ScheduledJob) (result *batch.ScheduledJob, err error) {
	result = &batch.ScheduledJob{}
	err = c.r.Put().Namespace(c.ns).Resource("scheduledJobs").Name(job.Name).SubResource("status").Body(job).Do().Into(result)
	return
}
