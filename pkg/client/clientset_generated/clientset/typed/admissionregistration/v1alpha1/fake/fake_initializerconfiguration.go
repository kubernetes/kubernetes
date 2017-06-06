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
	v1alpha1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1"
)

// FakeInitializerConfigurations implements InitializerConfigurationInterface
type FakeInitializerConfigurations struct {
	Fake *FakeAdmissionregistrationV1alpha1
}

var initializerconfigurationsResource = schema.GroupVersionResource{Group: "admissionregistration.k8s.io", Version: "v1alpha1", Resource: "initializerconfigurations"}

var initializerconfigurationsKind = schema.GroupVersionKind{Group: "admissionregistration.k8s.io", Version: "v1alpha1", Kind: "InitializerConfiguration"}

func (c *FakeInitializerConfigurations) Create(initializerConfiguration *v1alpha1.InitializerConfiguration) (result *v1alpha1.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(initializerconfigurationsResource, initializerConfiguration), &v1alpha1.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.InitializerConfiguration), err
}

func (c *FakeInitializerConfigurations) Update(initializerConfiguration *v1alpha1.InitializerConfiguration) (result *v1alpha1.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(initializerconfigurationsResource, initializerConfiguration), &v1alpha1.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.InitializerConfiguration), err
}

func (c *FakeInitializerConfigurations) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(initializerconfigurationsResource, name), &v1alpha1.InitializerConfiguration{})
	return err
}

func (c *FakeInitializerConfigurations) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(initializerconfigurationsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.InitializerConfigurationList{})
	return err
}

func (c *FakeInitializerConfigurations) Get(name string, options v1.GetOptions) (result *v1alpha1.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(initializerconfigurationsResource, name), &v1alpha1.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.InitializerConfiguration), err
}

func (c *FakeInitializerConfigurations) List(opts v1.ListOptions) (result *v1alpha1.InitializerConfigurationList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(initializerconfigurationsResource, initializerconfigurationsKind, opts), &v1alpha1.InitializerConfigurationList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.InitializerConfigurationList{}
	for _, item := range obj.(*v1alpha1.InitializerConfigurationList).Items {
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

// Patch applies the patch and returns the patched initializerConfiguration.
func (c *FakeInitializerConfigurations) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.InitializerConfiguration, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(initializerconfigurationsResource, name, data, subresources...), &v1alpha1.InitializerConfiguration{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.InitializerConfiguration), err
}
