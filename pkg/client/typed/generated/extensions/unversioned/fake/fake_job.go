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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeJobs implements JobInterface
type FakeJobs struct {
	Fake *FakeExtensions
	ns   string
}

func (c *FakeJobs) Create(job *extensions.Job) (result *extensions.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction("jobs", c.ns, job), &extensions.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Job), err
}

func (c *FakeJobs) Update(job *extensions.Job) (result *extensions.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction("jobs", c.ns, job), &extensions.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Job), err
}

func (c *FakeJobs) UpdateStatus(job *extensions.Job) (*extensions.Job, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction("jobs", "status", c.ns, job), &extensions.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Job), err
}

func (c *FakeJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction("jobs", c.ns, name), &extensions.Job{})

	return err
}

func (c *FakeJobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction("jobs", c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.JobList{})
	return err
}

func (c *FakeJobs) Get(name string) (result *extensions.Job, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction("jobs", c.ns, name), &extensions.Job{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Job), err
}

func (c *FakeJobs) List(opts api.ListOptions) (result *extensions.JobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction("jobs", c.ns, opts), &extensions.JobList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.JobList{}
	for _, item := range obj.(*extensions.JobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested jobs.
func (c *FakeJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction("jobs", c.ns, opts))

}
