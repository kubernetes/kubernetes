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
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
)

// FakeInitializerConfigurations implements InitializerConfigurationInterface
type FakeInitializerConfigurations struct {
	Fake *FakeAdmissionregistration
}

var initializerconfigurationsResource = schema.GroupVersionResource{Group: "admissionregistration.k8s.io", Version: "", Resource: "initializerconfigurations"}

var initializerconfigurationsKind = schema.GroupVersionKind{Group: "admissionregistration.k8s.io", Version: "", Kind: "InitializerConfiguration"}

// Get takes name of the initializerConfiguration, and returns the corresponding initializerConfiguration object, and an error if there is any.
func (c *FakeInitializerConfigurations) Get(name string, options v1.GetOptions) (result *admissionregistration.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(initializerconfigurationsResource, name), &admissionregistration.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.InitializerConfiguration), err
}

// List takes label and field selectors, and returns the list of InitializerConfigurations that match those selectors.
func (c *FakeInitializerConfigurations) List(opts v1.ListOptions) (result *admissionregistration.InitializerConfigurationList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(initializerconfigurationsResource, initializerconfigurationsKind, opts), &admissionregistration.InitializerConfigurationList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &admissionregistration.InitializerConfigurationList{}
	for _, item := range obj.(*admissionregistration.InitializerConfigurationList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested initializerConfigurations.
func (c *FakeInitializerConfigurations) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(initializerconfigurationsResource, opts))
}

// Create takes the representation of a initializerConfiguration and creates it.  Returns the server's representation of the initializerConfiguration, and an error, if there is any.
func (c *FakeInitializerConfigurations) Create(initializerConfiguration *admissionregistration.InitializerConfiguration) (result *admissionregistration.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(initializerconfigurationsResource, initializerConfiguration), &admissionregistration.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.InitializerConfiguration), err
}

// Update takes the representation of a initializerConfiguration and updates it. Returns the server's representation of the initializerConfiguration, and an error, if there is any.
func (c *FakeInitializerConfigurations) Update(initializerConfiguration *admissionregistration.InitializerConfiguration) (result *admissionregistration.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(initializerconfigurationsResource, initializerConfiguration), &admissionregistration.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.InitializerConfiguration), err
}

// Delete takes name of the initializerConfiguration and deletes it. Returns an error if one occurs.
func (c *FakeInitializerConfigurations) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(initializerconfigurationsResource, name), &admissionregistration.InitializerConfiguration{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeInitializerConfigurations) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(initializerconfigurationsResource, listOptions)

	_, err := c.Fake.Invokes(action, &admissionregistration.InitializerConfigurationList{})
	return err
}

// Patch applies the patch and returns the patched initializerConfiguration.
func (c *FakeInitializerConfigurations) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *admissionregistration.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(initializerconfigurationsResource, name, data, subresources...), &admissionregistration.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.InitializerConfiguration), err
}
