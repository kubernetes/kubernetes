/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	kclientlib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeDeployments implements DeploymentInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeDeployments struct {
	Fake      *FakeExperimental
	Namespace string
}

// Ensure statically that FakeDeployments implements DeploymentInterface.
var _ kclientlib.DeploymentInterface = &FakeDeployments{}

func (c *FakeDeployments) Get(name string) (*extensions.Deployment, error) {
	obj, err := c.Fake.Invokes(NewGetAction("deployments", c.Namespace, name), &extensions.Deployment{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) List(opts api.ListOptions) (*extensions.DeploymentList, error) {
	obj, err := c.Fake.Invokes(NewListAction("deployments", c.Namespace, opts), &extensions.DeploymentList{})
	if obj == nil {
		return nil, err
	}
	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.DeploymentList{}
	for _, deployment := range obj.(*extensions.DeploymentList).Items {
		if label.Matches(labels.Set(deployment.Labels)) {
			list.Items = append(list.Items, deployment)
		}
	}
	return list, err
}

func (c *FakeDeployments) Create(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("deployments", c.Namespace, deployment), deployment)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) Update(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("deployments", c.Namespace, deployment), deployment)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) UpdateStatus(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("deployments", "status", c.Namespace, deployment), deployment)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("deployments", c.Namespace, name), &extensions.Deployment{})
	return err
}

func (c *FakeDeployments) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("deployments", c.Namespace, opts))
}

func (c *FakeDeployments) Rollback(deploymentRollback *extensions.DeploymentRollback) error {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "deployments"
	action.Subresource = "rollback"
	action.Object = deploymentRollback

	_, err := c.Fake.Invokes(action, deploymentRollback)
	return err
}
