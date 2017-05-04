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
	apiextensions "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
)

// FakeCustomResources implements CustomResourceInterface
type FakeCustomResources struct {
	Fake *FakeApiextensions
}

var customresourcesResource = schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "", Resource: "customresources"}

var customresourcesKind = schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "", Kind: "CustomResource"}

func (c *FakeCustomResources) Create(customResource *apiextensions.CustomResource) (result *apiextensions.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(customresourcesResource, customResource), &apiextensions.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResource), err
}

func (c *FakeCustomResources) Update(customResource *apiextensions.CustomResource) (result *apiextensions.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(customresourcesResource, customResource), &apiextensions.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResource), err
}

func (c *FakeCustomResources) UpdateStatus(customResource *apiextensions.CustomResource) (*apiextensions.CustomResource, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(customresourcesResource, "status", customResource), &apiextensions.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResource), err
}

func (c *FakeCustomResources) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(customresourcesResource, name), &apiextensions.CustomResource{})
	return err
}

func (c *FakeCustomResources) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(customresourcesResource, listOptions)

	_, err := c.Fake.Invokes(action, &apiextensions.CustomResourceList{})
	return err
}

func (c *FakeCustomResources) Get(name string, options v1.GetOptions) (result *apiextensions.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(customresourcesResource, name), &apiextensions.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResource), err
}

func (c *FakeCustomResources) List(opts v1.ListOptions) (result *apiextensions.CustomResourceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(customresourcesResource, customresourcesKind, opts), &apiextensions.CustomResourceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &apiextensions.CustomResourceList{}
	for _, item := range obj.(*apiextensions.CustomResourceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested customResources.
func (c *FakeCustomResources) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(customresourcesResource, opts))
}

// Patch applies the patch and returns the patched customResource.
func (c *FakeCustomResources) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apiextensions.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(customresourcesResource, name, data, subresources...), &apiextensions.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResource), err
}
