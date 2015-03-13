/*
Copyright 2014 Google Inc. All rights reserved.

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
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-resourceQuotas"})
	return api.Scheme.CopyOrDie(&c.Fake.ResourceQuotasList).(*api.ResourceQuotaList), nil
}

func (c *FakeResourceQuotas) Get(name string) (*api.ResourceQuota, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-resourceQuota", Value: name})
	return &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: name, Namespace: c.Namespace}}, nil
}

func (c *FakeResourceQuotas) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-resourceQuota", Value: name})
	return nil
}

func (c *FakeResourceQuotas) Create(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-resourceQuota"})
	return &api.ResourceQuota{}, nil
}

func (c *FakeResourceQuotas) Update(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-resourceQuota", Value: resourceQuota.Name})
	return &api.ResourceQuota{}, nil
}

func (c *FakeResourceQuotas) Status(resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-status-resourceQuota", Value: resourceQuota.Name})
	c.Fake.ResourceQuotaStatus = *resourceQuota
	return &api.ResourceQuota{}, nil
}

func (c *FakeResourceQuotas) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-resourceQuota", Value: resourceVersion})
	return c.Fake.Watch, nil
}
