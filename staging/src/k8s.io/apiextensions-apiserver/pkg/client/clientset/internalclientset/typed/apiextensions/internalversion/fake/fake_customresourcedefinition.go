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
	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeCustomResourceDefinitions implements CustomResourceDefinitionInterface
type FakeCustomResourceDefinitions struct {
	Fake *FakeApiextensions
}

var customresourcedefinitionsResource = schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "", Resource: "customresourcedefinitions"}

var customresourcedefinitionsKind = schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "", Kind: "CustomResourceDefinition"}

// Get takes name of the customResourceDefinition, and returns the corresponding customResourceDefinition object, and an error if there is any.
func (c *FakeCustomResourceDefinitions) Get(name string, options v1.GetOptions) (result *apiextensions.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(customresourcedefinitionsResource, name), &apiextensions.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResourceDefinition), err
}

// List takes label and field selectors, and returns the list of CustomResourceDefinitions that match those selectors.
func (c *FakeCustomResourceDefinitions) List(opts v1.ListOptions) (result *apiextensions.CustomResourceDefinitionList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(customresourcedefinitionsResource, customresourcedefinitionsKind, opts), &apiextensions.CustomResourceDefinitionList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &apiextensions.CustomResourceDefinitionList{}
	for _, item := range obj.(*apiextensions.CustomResourceDefinitionList).Items {
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

// Create takes the representation of a customResourceDefinition and creates it.  Returns the server's representation of the customResourceDefinition, and an error, if there is any.
func (c *FakeCustomResourceDefinitions) Create(customResourceDefinition *apiextensions.CustomResourceDefinition) (result *apiextensions.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(customresourcedefinitionsResource, customResourceDefinition), &apiextensions.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResourceDefinition), err
}

// Update takes the representation of a customResourceDefinition and updates it. Returns the server's representation of the customResourceDefinition, and an error, if there is any.
func (c *FakeCustomResourceDefinitions) Update(customResourceDefinition *apiextensions.CustomResourceDefinition) (result *apiextensions.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(customresourcedefinitionsResource, customResourceDefinition), &apiextensions.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResourceDefinition), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeCustomResourceDefinitions) UpdateStatus(customResourceDefinition *apiextensions.CustomResourceDefinition) (*apiextensions.CustomResourceDefinition, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(customresourcedefinitionsResource, "status", customResourceDefinition), &apiextensions.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResourceDefinition), err
}

// Delete takes name of the customResourceDefinition and deletes it. Returns an error if one occurs.
func (c *FakeCustomResourceDefinitions) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(customresourcedefinitionsResource, name), &apiextensions.CustomResourceDefinition{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeCustomResourceDefinitions) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(customresourcedefinitionsResource, listOptions)

	_, err := c.Fake.Invokes(action, &apiextensions.CustomResourceDefinitionList{})
	return err
}

// Patch applies the patch and returns the patched customResourceDefinition.
func (c *FakeCustomResourceDefinitions) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apiextensions.CustomResourceDefinition, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(customresourcedefinitionsResource, name, data, subresources...), &apiextensions.CustomResourceDefinition{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apiextensions.CustomResourceDefinition), err
}
