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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeCronJobs implements CronJobInterface
type FakeCronJobs struct {
	Fake *FakeBatchV2alpha1
	ns   string
}

var cronjobsResource = schema.GroupVersionResource{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"}

func (c *FakeCronJobs) Create(cronJob *v2alpha1.CronJob) (result *v2alpha1.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(cronjobsResource, c.ns, cronJob), &v2alpha1.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.CronJob), err
}

func (c *FakeCronJobs) Update(cronJob *v2alpha1.CronJob) (result *v2alpha1.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(cronjobsResource, c.ns, cronJob), &v2alpha1.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.CronJob), err
}

func (c *FakeCronJobs) UpdateStatus(cronJob *v2alpha1.CronJob) (*v2alpha1.CronJob, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(cronjobsResource, "status", c.ns, cronJob), &v2alpha1.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.CronJob), err
}

func (c *FakeCronJobs) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(cronjobsResource, c.ns, name), &v2alpha1.CronJob{})

	return err
}

func (c *FakeCronJobs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(cronjobsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v2alpha1.CronJobList{})
	return err
}

func (c *FakeCronJobs) Get(name string) (result *v2alpha1.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(cronjobsResource, c.ns, name), &v2alpha1.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.CronJob), err
}

func (c *FakeCronJobs) List(opts v1.ListOptions) (result *v2alpha1.CronJobList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(cronjobsResource, c.ns, opts), &v2alpha1.CronJobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v2alpha1.CronJobList{}
	for _, item := range obj.(*v2alpha1.CronJobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested cronJobs.
func (c *FakeCronJobs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(cronjobsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched cronJob.
func (c *FakeCronJobs) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v2alpha1.CronJob, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(cronjobsResource, c.ns, name, data, subresources...), &v2alpha1.CronJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v2alpha1.CronJob), err
}
