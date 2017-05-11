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
	v1alpha1 "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1"
)

// FakeCustomResources implements CustomResourceInterface
type FakeCustomResources struct {
	Fake *FakeApiextensionsV1alpha1
}

var customresourcesResource = schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "v1alpha1", Resource: "customresources"}

var customresourcesKind = schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "v1alpha1", Kind: "CustomResource"}

func (c *FakeCustomResources) Create(customResource *v1alpha1.CustomResource) (result *v1alpha1.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(customresourcesResource, customResource), &v1alpha1.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CustomResource), err
}

func (c *FakeCustomResources) Update(customResource *v1alpha1.CustomResource) (result *v1alpha1.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(customresourcesResource, customResource), &v1alpha1.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CustomResource), err
}

func (c *FakeCustomResources) UpdateStatus(customResource *v1alpha1.CustomResource) (*v1alpha1.CustomResource, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(customresourcesResource, "status", customResource), &v1alpha1.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CustomResource), err
}

func (c *FakeCustomResources) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(customresourcesResource, name), &v1alpha1.CustomResource{})
	return err
}

func (c *FakeCustomResources) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(customresourcesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.CustomResourceList{})
	return err
}

func (c *FakeCustomResources) Get(name string, options v1.GetOptions) (result *v1alpha1.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(customresourcesResource, name), &v1alpha1.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CustomResource), err
}

func (c *FakeCustomResources) List(opts v1.ListOptions) (result *v1alpha1.CustomResourceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(customresourcesResource, customresourcesKind, opts), &v1alpha1.CustomResourceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.CustomResourceList{}
	for _, item := range obj.(*v1alpha1.CustomResourceList).Items {
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
func (c *FakeCustomResources) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.CustomResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(customresourcesResource, name, data, subresources...), &v1alpha1.CustomResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CustomResource), err
}
