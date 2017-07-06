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
	v1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeCustomResourceDefinitions implements CustomResourceDefinitionInterface
type FakeCustomResourceDefinitions struct {
	Fake *FakeApiextensionsV1beta1
}

var customresourcedefinitionsResource = schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "v1beta1", Resource: "customresourcedefinitions"}

var customresourcedefinitionsKind = schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "v1beta1", Kind: "CustomResourceDefinition"}

func (c *FakeCustomResourceDefinitions) Create(customResourceDefinition *v1beta1.CustomResourceDefinition) (result *v1beta1.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(customresourcedefinitionsResource, customResourceDefinition), &v1beta1.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.CustomResourceDefinition), err
}

func (c *FakeCustomResourceDefinitions) Update(customResourceDefinition *v1beta1.CustomResourceDefinition) (result *v1beta1.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(customresourcedefinitionsResource, customResourceDefinition), &v1beta1.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.CustomResourceDefinition), err
}

func (c *FakeCustomResourceDefinitions) UpdateStatus(customResourceDefinition *v1beta1.CustomResourceDefinition) (*v1beta1.CustomResourceDefinition, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(customresourcedefinitionsResource, "status", customResourceDefinition), &v1beta1.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.CustomResourceDefinition), err
}

func (c *FakeCustomResourceDefinitions) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(customresourcedefinitionsResource, name), &v1beta1.CustomResourceDefinition{})
	return err
}

func (c *FakeCustomResourceDefinitions) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(customresourcedefinitionsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.CustomResourceDefinitionList{})
	return err
}

func (c *FakeCustomResourceDefinitions) Get(name string, options v1.GetOptions) (result *v1beta1.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(customresourcedefinitionsResource, name), &v1beta1.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.CustomResourceDefinition), err
}

func (c *FakeCustomResourceDefinitions) List(opts v1.ListOptions) (result *v1beta1.CustomResourceDefinitionList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(customresourcedefinitionsResource, customresourcedefinitionsKind, opts), &v1beta1.CustomResourceDefinitionList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.CustomResourceDefinitionList{}
	for _, item := range obj.(*v1beta1.CustomResourceDefinitionList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested customResourceDefinitions.
func (c *FakeCustomResourceDefinitions) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(customresourcedefinitionsResource, opts))
}

// Patch applies the patch and returns the patched customResourceDefinition.
func (c *FakeCustomResourceDefinitions) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(customresourcedefinitionsResource, name, data, subresources...), &v1beta1.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.CustomResourceDefinition), err
}
