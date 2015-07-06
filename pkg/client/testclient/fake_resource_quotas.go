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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeResourceQuotas implements ResourceQuotaInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeResourceQuotas struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeResourceQuotas) List(selector labels.Selector) (*api.ResourceQuotaList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-resourceQuotas"}, &api.ResourceQuotaList{})
	return obj.(*api.ResourceQuotaList), err
}

func (c *FakeResourceQuotas) Get(name string) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-resourceQuota", Value: name}, &api.ResourceQuota{})
	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-resourceQuota", Value: name}, &api.ResourceQuota{})
	return err
}

func (c *FakeResourceQuotas) Create(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-resourceQuota", Value: resourceQuota}, &api.ResourceQuota{})
	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) Update(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-resourceQuota", Value: resourceQuota}, &api.ResourceQuota{})
	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) UpdateStatus(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-resourceQuota", Value: resourceQuota}, &api.ResourceQuota{})
	return obj.(*api.ResourceQuota), err
}

func (c *FakeResourceQuotas) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(FakeAction{Action: "watch-resourceQuota", Value: resourceVersion}, nil)
	return c.Fake.Watch, nil
}
