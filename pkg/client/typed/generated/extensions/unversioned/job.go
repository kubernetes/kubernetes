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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// JobNamespacer has methods to work with Job resources in a namespace
type JobNamespacer interface {
	Jobs(namespace string) JobInterface
}

// JobInterface has methods to work with Job resources.
type JobInterface interface {
	Create(*extensions.Job) (*extensions.Job, error)
	Update(*extensions.Job) (*extensions.Job, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*extensions.Job, error)
	List(opts unversioned.ListOptions) (*extensions.JobList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
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
func (c *jobs) Create(job *extensions.Job) (result *extensions.Job, err error) {
	result = &extensions.Job{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("jobs").
		Body(job).
		Do().
		Into(result)
	return
}

// Update takes the representation of a job and updates it. Returns the server's representation of the job, and an error, if there is any.
func (c *jobs) Update(job *extensions.Job) (result *extensions.Job, err error) {
	result = &extensions.Job{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("jobs").
		Name(job.Name).
		Body(job).
		Do().
		Into(result)
	return
}

// Delete takes name of the job and deletes it. Returns an error if one occurs.
func (c *jobs) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("jobs").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("jobs").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the job, and returns the corresponding job object, and an error if there is any.
func (c *jobs) Get(name string) (result *extensions.Job, err error) {
	result = &extensions.Job{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("jobs").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Jobs that match those selectors.
func (c *jobs) List(opts unversioned.ListOptions) (result *extensions.JobList, err error) {
	result = &extensions.JobList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *jobs) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("jobs").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
