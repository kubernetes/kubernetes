/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// JobsNamespacer has methods to work with Job resources in a namespace
type JobsNamespacer interface {
	Jobs(namespace string) JobInterface
}

// JobInterface has methods to work with Job resources.
type JobInterface interface {
	List(selector labels.Selector) (*api.JobList, error)
	Get(name string) (*api.Job, error)
	Create(ctrl *api.Job) (*api.Job, error)
	Update(ctrl *api.Job) (*api.Job, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// jobs implements JobsNamespacer interface
type jobs struct {
	r  *Client
	ns string
}

// newJobs returns a PodsClient
func newJobs(c *Client, namespace string) *jobs {
	return &jobs{c, namespace}
}

// List takes a selector, and returns the list of replication controllers that match that selector.
func (c *jobs) List(selector labels.Selector) (result *api.JobList, err error) {
	result = &api.JobList{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").LabelsSelectorParam(selector).Do().Into(result)
	return
}

// Get returns information about a particular replication controller.
func (c *jobs) Get(name string) (result *api.Job, err error) {
	result = &api.Job{}
	err = c.r.Get().Namespace(c.ns).Resource("jobs").Name(name).Do().Into(result)
	return
}

// Create creates a new replication controller.
func (c *jobs) Create(controller *api.Job) (result *api.Job, err error) {
	result = &api.Job{}
	err = c.r.Post().Namespace(c.ns).Resource("jobs").Body(controller).Do().Into(result)
	return
}

// Update updates an existing replication controller.
func (c *jobs) Update(controller *api.Job) (result *api.Job, err error) {
	result = &api.Job{}
	err = c.r.Put().Namespace(c.ns).Resource("jobs").Name(controller.Name).Body(controller).Do().Into(result)
	return
}

// Delete deletes an existing replication controller.
func (c *jobs) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("jobs").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested controllers.
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
