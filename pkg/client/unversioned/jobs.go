/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// JobsNamespacer has methods to work with Job resources in a namespace
type JobsNamespacer interface {
	Jobs(namespace string) JobInterface
}

// JobInterface exposes methods to work on Job resources.
type JobInterface interface {
	List(label labels.Selector) (*expapi.JobList, error)
	Get(name string) (*expapi.Job, error)
	Create(job *expapi.Job) (*expapi.Job, error)
	Update(job *expapi.Job) (*expapi.Job, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// jobs implements JobsNamespacer interface
type jobs struct {
	r  *Client
	ns string
}

// newJobs returns a jobs
func newJobs(c *Client, namespace string) *jobs {
	return &jobs{c, namespace}
}

// List returns a list of jobs that match the label and field selectors.
func (c *jobs) List(label labels.Selector) (result *expapi.JobList, err error) {
	result = &expapi.JobList{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").LabelsSelectorParam(label).Do().Into(result)
	return
}

// Get returns information about a particular job.
func (c *jobs) Get(name string) (result *expapi.Job, err error) {
	result = &expapi.Job{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").Name(name).Do().Into(result)
	return
}

// Create creates a new job.
func (c *jobs) Create(job *expapi.Job) (result *expapi.Job, err error) {
	result = &expapi.Job{}
	err = c.r.Post().Namespace(c.ns).Resource("jobs").Body(job).Do().Into(result)
	return
}

// Update updates an existing job.
func (c *jobs) Update(job *expapi.Job) (result *expapi.Job, err error) {
	result = &expapi.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(job.Name).Body(job).Do().Into(result)
	return
}

// Delete deletes a job, returns error if one occurs.
func (c *jobs) Delete(name string) (err error) {
	err = c.r.Delete().Namespace(c.ns).Resource("jobs").Name(name).Do().Error()
	return
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *jobs) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("jobs").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
