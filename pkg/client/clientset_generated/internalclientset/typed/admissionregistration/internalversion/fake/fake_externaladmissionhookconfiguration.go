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

// FakeExternalAdmissionHookConfigurations implements ExternalAdmissionHookConfigurationInterface
type FakeExternalAdmissionHookConfigurations struct {
	Fake *FakeAdmissionregistration
}

var externaladmissionhookconfigurationsResource = schema.GroupVersionResource{Group: "admissionregistration.k8s.io", Version: "", Resource: "externaladmissionhookconfigurations"}

var externaladmissionhookconfigurationsKind = schema.GroupVersionKind{Group: "admissionregistration.k8s.io", Version: "", Kind: "ExternalAdmissionHookConfiguration"}

// Get takes name of the externalAdmissionHookConfiguration, and returns the corresponding externalAdmissionHookConfiguration object, and an error if there is any.
func (c *FakeExternalAdmissionHookConfigurations) Get(name string, options v1.GetOptions) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(externaladmissionhookconfigurationsResource, name), &admissionregistration.ExternalAdmissionHookConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.ExternalAdmissionHookConfiguration), err
}

// List takes label and field selectors, and returns the list of ExternalAdmissionHookConfigurations that match those selectors.
func (c *FakeExternalAdmissionHookConfigurations) List(opts v1.ListOptions) (result *admissionregistration.ExternalAdmissionHookConfigurationList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(externaladmissionhookconfigurationsResource, externaladmissionhookconfigurationsKind, opts), &admissionregistration.ExternalAdmissionHookConfigurationList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &admissionregistration.ExternalAdmissionHookConfigurationList{}
	for _, item := range obj.(*admissionregistration.ExternalAdmissionHookConfigurationList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested externalAdmissionHookConfigurations.
func (c *FakeExternalAdmissionHookConfigurations) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(externaladmissionhookconfigurationsResource, opts))
}

// Create takes the representation of a externalAdmissionHookConfiguration and creates it.  Returns the server's representation of the externalAdmissionHookConfiguration, and an error, if there is any.
func (c *FakeExternalAdmissionHookConfigurations) Create(externalAdmissionHookConfiguration *admissionregistration.ExternalAdmissionHookConfiguration) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(externaladmissionhookconfigurationsResource, externalAdmissionHookConfiguration), &admissionregistration.ExternalAdmissionHookConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.ExternalAdmissionHookConfiguration), err
}

// Update takes the representation of a externalAdmissionHookConfiguration and updates it. Returns the server's representation of the externalAdmissionHookConfiguration, and an error, if there is any.
func (c *FakeExternalAdmissionHookConfigurations) Update(externalAdmissionHookConfiguration *admissionregistration.ExternalAdmissionHookConfiguration) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(externaladmissionhookconfigurationsResource, externalAdmissionHookConfiguration), &admissionregistration.ExternalAdmissionHookConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.ExternalAdmissionHookConfiguration), err
}

// Delete takes name of the externalAdmissionHookConfiguration and deletes it. Returns an error if one occurs.
func (c *FakeExternalAdmissionHookConfigurations) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(externaladmissionhookconfigurationsResource, name), &admissionregistration.ExternalAdmissionHookConfiguration{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeExternalAdmissionHookConfigurations) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(externaladmissionhookconfigurationsResource, listOptions)

	_, err := c.Fake.Invokes(action, &admissionregistration.ExternalAdmissionHookConfigurationList{})
	return err
}

// Patch applies the patch and returns the patched externalAdmissionHookConfiguration.
func (c *FakeExternalAdmissionHookConfigurations) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(externaladmissionhookconfigurationsResource, name, data, subresources...), &admissionregistration.ExternalAdmissionHookConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*admissionregistration.ExternalAdmissionHookConfiguration), err
}
