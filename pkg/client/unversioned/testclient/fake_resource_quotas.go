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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	resourceQuotaResourceName string = "resourceQuotas"
)

// FakeResourceQuotas implements ResourceQuotaInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeResourceQuotas struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeResourceQuotas) Get(name string) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(NewGetAction(resourceQuotaResourceName, c.Namespace, name), &api.ResourceQuota{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) List(opts api.ListOptions) (*api.ResourceQuotaList, error) {
	obj, err := c.Fake.Invokes(NewListAction(resourceQuotaResourceName, c.Namespace, opts), &api.ResourceQuotaList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ResourceQuotaList), err
}

func (c *FakeResourceQuotas) Create(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(NewCreateAction(resourceQuotaResourceName, c.Namespace, resourceQuota), resourceQuota)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) Update(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction(resourceQuotaResourceName, c.Namespace, resourceQuota), resourceQuota)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction(resourceQuotaResourceName, c.Namespace, name), &api.ResourceQuota{})
	return err
}

func (c *FakeResourceQuotas) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction(resourceQuotaResourceName, c.Namespace, opts))
}

func (c *FakeResourceQuotas) UpdateStatus(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction(resourceQuotaResourceName, "status", c.Namespace, resourceQuota), resourceQuota)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ResourceQuota), err
}
