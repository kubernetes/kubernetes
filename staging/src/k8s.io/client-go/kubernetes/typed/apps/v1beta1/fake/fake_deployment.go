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
	v1beta1 "k8s.io/client-go/pkg/apis/apps/v1beta1"
	testing "k8s.io/client-go/testing"
)

// FakeDeployments implements DeploymentInterface
type FakeDeployments struct {
	Fake *FakeAppsV1beta1
	ns   string
}

var deploymentsResource = schema.GroupVersionResource{Group: "apps", Version: "v1beta1", Resource: "deployments"}

func (c *FakeDeployments) Create(deployment *v1beta1.Deployment) (result *v1beta1.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deploymentsResource, c.ns, deployment), &v1beta1.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Deployment), err
}

func (c *FakeDeployments) Update(deployment *v1beta1.Deployment) (result *v1beta1.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deploymentsResource, c.ns, deployment), &v1beta1.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Deployment), err
}

func (c *FakeDeployments) UpdateStatus(deployment *v1beta1.Deployment) (*v1beta1.Deployment, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(deploymentsResource, "status", c.ns, deployment), &v1beta1.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Deployment), err
}

func (c *FakeDeployments) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deploymentsResource, c.ns, name), &v1beta1.Deployment{})

	return err
}

func (c *FakeDeployments) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deploymentsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.DeploymentList{})
	return err
}

func (c *FakeDeployments) Get(name string, options v1.GetOptions) (result *v1beta1.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deploymentsResource, c.ns, name), &v1beta1.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Deployment), err
}

func (c *FakeDeployments) List(opts v1.ListOptions) (result *v1beta1.DeploymentList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deploymentsResource, c.ns, opts), &v1beta1.DeploymentList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.DeploymentList{}
	for _, item := range obj.(*v1beta1.DeploymentList).Items {
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
func (c *FakeDeployments) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Deployment, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deploymentsResource, c.ns, name, data, subresources...), &v1beta1.Deployment{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Deployment), err
}
