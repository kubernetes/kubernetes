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
	v1beta1 "k8s.io/api/extensions/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeThirdPartyResources implements ThirdPartyResourceInterface
type FakeThirdPartyResources struct {
	Fake *FakeExtensionsV1beta1
}

var thirdpartyresourcesResource = schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "thirdpartyresources"}

var thirdpartyresourcesKind = schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "ThirdPartyResource"}

// Get takes name of the thirdPartyResource, and returns the corresponding thirdPartyResource object, and an error if there is any.
func (c *FakeThirdPartyResources) Get(name string, options v1.GetOptions) (result *v1beta1.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(thirdpartyresourcesResource, name), &v1beta1.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ThirdPartyResource), err
}

// List takes label and field selectors, and returns the list of ThirdPartyResources that match those selectors.
func (c *FakeThirdPartyResources) List(opts v1.ListOptions) (result *v1beta1.ThirdPartyResourceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(thirdpartyresourcesResource, thirdpartyresourcesKind, opts), &v1beta1.ThirdPartyResourceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.ThirdPartyResourceList{}
	for _, item := range obj.(*v1beta1.ThirdPartyResourceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested thirdPartyResources.
func (c *FakeThirdPartyResources) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(thirdpartyresourcesResource, opts))
}

// Create takes the representation of a thirdPartyResource and creates it.  Returns the server's representation of the thirdPartyResource, and an error, if there is any.
func (c *FakeThirdPartyResources) Create(thirdPartyResource *v1beta1.ThirdPartyResource) (result *v1beta1.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(thirdpartyresourcesResource, thirdPartyResource), &v1beta1.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ThirdPartyResource), err
}

// Update takes the representation of a thirdPartyResource and updates it. Returns the server's representation of the thirdPartyResource, and an error, if there is any.
func (c *FakeThirdPartyResources) Update(thirdPartyResource *v1beta1.ThirdPartyResource) (result *v1beta1.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(thirdpartyresourcesResource, thirdPartyResource), &v1beta1.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ThirdPartyResource), err
}

// Delete takes name of the thirdPartyResource and deletes it. Returns an error if one occurs.
func (c *FakeThirdPartyResources) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(thirdpartyresourcesResource, name), &v1beta1.ThirdPartyResource{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeThirdPartyResources) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(thirdpartyresourcesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.ThirdPartyResourceList{})
	return err
}

// Patch applies the patch and returns the patched thirdPartyResource.
func (c *FakeThirdPartyResources) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(thirdpartyresourcesResource, name, data, subresources...), &v1beta1.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.ThirdPartyResource), err
}
