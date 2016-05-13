/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// FakeThirdPartyResourceDatas implements ThirdPartyResourceDataInterface
type FakeThirdPartyResourceDatas struct {
	Fake *FakeExtensions
	ns   string
}

var thirdpartyresourcedatasResource = unversioned.GroupVersionResource{Group: "extensions", Version: "", Resource: "thirdpartyresourcedatas"}

func (c *FakeThirdPartyResourceDatas) Create(thirdPartyResourceData *extensions.ThirdPartyResourceData) (result *extensions.ThirdPartyResourceData, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(thirdpartyresourcedatasResource, c.ns, thirdPartyResourceData), &extensions.ThirdPartyResourceData{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceData), err
}

func (c *FakeThirdPartyResourceDatas) Update(thirdPartyResourceData *extensions.ThirdPartyResourceData) (result *extensions.ThirdPartyResourceData, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(thirdpartyresourcedatasResource, c.ns, thirdPartyResourceData), &extensions.ThirdPartyResourceData{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceData), err
}

func (c *FakeThirdPartyResourceDatas) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(thirdpartyresourcedatasResource, c.ns, name), &extensions.ThirdPartyResourceData{})

	return err
}

func (c *FakeThirdPartyResourceDatas) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(thirdpartyresourcedatasResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.ThirdPartyResourceDataList{})
	return err
}

func (c *FakeThirdPartyResourceDatas) Get(name string) (result *extensions.ThirdPartyResourceData, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(thirdpartyresourcedatasResource, c.ns, name), &extensions.ThirdPartyResourceData{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceData), err
}

func (c *FakeThirdPartyResourceDatas) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceDataList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(thirdpartyresourcedatasResource, c.ns, opts), &extensions.ThirdPartyResourceDataList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.ThirdPartyResourceDataList{}
	for _, item := range obj.(*extensions.ThirdPartyResourceDataList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested thirdPartyResourceDatas.
func (c *FakeThirdPartyResourceDatas) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(thirdpartyresourcedatasResource, c.ns, opts))

}
