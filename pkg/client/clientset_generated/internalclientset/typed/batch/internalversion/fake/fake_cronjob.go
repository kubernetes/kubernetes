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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	batch "k8s.io/kubernetes/pkg/apis/batch"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeCronJobs implements CronJobInterface
type FakeCronJobs struct {
	Fake *FakeBatch
	ns   string
}

var cronjobsResource = unversioned.GroupVersionResource{Group: "batch", Version: "", Resource: "cronjobs"}

func (c *FakeCronJobs) Create(cronJob *batch.CronJob) (result *batch.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(cronjobsResource, c.ns, cronJob), &batch.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.CronJob), err
}

func (c *FakeCronJobs) Update(cronJob *batch.CronJob) (result *batch.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(cronjobsResource, c.ns, cronJob), &batch.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.CronJob), err
}

func (c *FakeCronJobs) UpdateStatus(cronJob *batch.CronJob) (*batch.CronJob, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(cronjobsResource, "status", c.ns, cronJob), &batch.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.CronJob), err
}

func (c *FakeCronJobs) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(cronjobsResource, c.ns, name), &batch.CronJob{})

	return err
}

func (c *FakeCronJobs) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(cronjobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &batch.CronJobList{})
	return err
}

func (c *FakeCronJobs) Get(name string) (result *batch.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(cronjobsResource, c.ns, name), &batch.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.CronJob), err
}

func (c *FakeCronJobs) List(opts api.ListOptions) (result *batch.CronJobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(cronjobsResource, c.ns, opts), &batch.CronJobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &batch.CronJobList{}
	for _, item := range obj.(*batch.CronJobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested cronJobs.
func (c *FakeCronJobs) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(cronjobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched cronJob.
func (c *FakeCronJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *batch.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(cronjobsResource, c.ns, name, data, subresources...), &batch.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*batch.CronJob), err
}
