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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeDeployments implements DeploymentsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeDeployments struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeDeployments) Get(name string) (*expapi.Deployment, error) {
	obj, err := c.Fake.Invokes(NewGetAction("deployments", c.Namespace, name), &expapi.Deployment{})
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.Deployment), err
}

func (c *FakeDeployments) List(label labels.Selector, field fields.Selector) (*expapi.DeploymentList, error) {
	obj, err := c.Fake.Invokes(NewListAction("deployments", c.Namespace, label, field), &expapi.DeploymentList{})
	if obj == nil {
		return nil, err
	}
	list := &expapi.DeploymentList{}
	for _, deployment := range obj.(*expapi.DeploymentList).Items {
		if label.Matches(labels.Set(deployment.Labels)) {
			list.Items = append(list.Items, deployment)
		}
	}
	return list, err
}

func (c *FakeDeployments) Create(deployment *expapi.Deployment) (*expapi.Deployment, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("deployments", c.Namespace, deployment), deployment)
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.Deployment), err
}

func (c *FakeDeployments) Update(deployment *expapi.Deployment) (*expapi.Deployment, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("deployments", c.Namespace, deployment), deployment)
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.Deployment), err
}

func (c *FakeDeployments) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("deployments", c.Namespace, name), &expapi.Deployment{})
	return err
}

func (c *FakeDeployments) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("deployments", c.Namespace, label, field, resourceVersion))
}
