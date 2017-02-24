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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
)

// FakeDeployments implements DeploymentInterface
type FakeDeployments struct {
	Fake *FakeExtensions
	ns   string
}

var deploymentsResource = schema.GroupVersionResource{Group: "extensions", Version: "", Resource: "deployments"}

func (c *FakeDeployments) Create(deployment *extensions.Deployment) (result *extensions.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deploymentsResource, c.ns, deployment), &extensions.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) Update(deployment *extensions.Deployment) (result *extensions.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deploymentsResource, c.ns, deployment), &extensions.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) UpdateStatus(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(deploymentsResource, "status", c.ns, deployment), &extensions.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deploymentsResource, c.ns, name), &extensions.Deployment{})

	return err
}

func (c *FakeDeployments) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deploymentsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.DeploymentList{})
	return err
}

func (c *FakeDeployments) Get(name string, options v1.GetOptions) (result *extensions.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deploymentsResource, c.ns, name), &extensions.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), err
}

func (c *FakeDeployments) List(opts v1.ListOptions) (result *extensions.DeploymentList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deploymentsResource, c.ns, opts), &extensions.DeploymentList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.DeploymentList{}
	for _, item := range obj.(*extensions.DeploymentList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested deployments.
func (c *FakeDeployments) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(deploymentsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched deployment.
func (c *FakeDeployments) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *extensions.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deploymentsResource, c.ns, name, data, subresources...), &extensions.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), err
}
