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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeThirdPartyResources implements ThirdPartyResourceInterface
type FakeThirdPartyResources struct {
	Fake *FakeExtensions
}

var thirdpartyresourcesResource = unversioned.GroupVersionResource{Group: "extensions", Version: "", Resource: "thirdpartyresources"}

func (c *FakeThirdPartyResources) Create(thirdPartyResource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(thirdpartyresourcesResource, thirdPartyResource), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) Update(thirdPartyResource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(thirdpartyresourcesResource, thirdPartyResource), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(thirdpartyresourcesResource, name), &extensions.ThirdPartyResource{})
	return err
}

func (c *FakeThirdPartyResources) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(thirdpartyresourcesResource, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.ThirdPartyResourceList{})
	return err
}

func (c *FakeThirdPartyResources) Get(name string) (result *extensions.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(thirdpartyresourcesResource, name), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(thirdpartyresourcesResource, opts), &extensions.ThirdPartyResourceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.ThirdPartyResourceList{}
	for _, item := range obj.(*extensions.ThirdPartyResourceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested thirdPartyResources.
func (c *FakeThirdPartyResources) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(thirdpartyresourcesResource, opts))
}

// Patch applies the patch and returns the patched thirdPartyResource.
func (c *FakeThirdPartyResources) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.ThirdPartyResource, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(thirdpartyresourcesResource, name, data, subresources...), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}
