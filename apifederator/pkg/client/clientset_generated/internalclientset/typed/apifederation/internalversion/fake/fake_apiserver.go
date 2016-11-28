/*
Copyright 2016 The Kubernetes Authors.

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
	apifederation "k8s.io/kubernetes/apifederator/pkg/apis/apifederation"
	api "k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeAPIServers implements APIServerInterface
type FakeAPIServers struct {
	Fake *FakeApifederation
}

var apiserversResource = schema.GroupVersionResource{Group: "apifederation.k8s.io", Version: "", Resource: "apiservers"}

func (c *FakeAPIServers) Create(aPIServer *apifederation.APIServer) (result *apifederation.APIServer, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(apiserversResource, aPIServer), &apifederation.APIServer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apifederation.APIServer), err
}

func (c *FakeAPIServers) Update(aPIServer *apifederation.APIServer) (result *apifederation.APIServer, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(apiserversResource, aPIServer), &apifederation.APIServer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apifederation.APIServer), err
}

func (c *FakeAPIServers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(apiserversResource, name), &apifederation.APIServer{})
	return err
}

func (c *FakeAPIServers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(apiserversResource, listOptions)

	_, err := c.Fake.Invokes(action, &apifederation.APIServerList{})
	return err
}

func (c *FakeAPIServers) Get(name string) (result *apifederation.APIServer, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(apiserversResource, name), &apifederation.APIServer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apifederation.APIServer), err
}

func (c *FakeAPIServers) List(opts api.ListOptions) (result *apifederation.APIServerList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(apiserversResource, opts), &apifederation.APIServerList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &apifederation.APIServerList{}
	for _, item := range obj.(*apifederation.APIServerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested aPIServers.
func (c *FakeAPIServers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(apiserversResource, opts))
}

// Patch applies the patch and returns the patched aPIServer.
func (c *FakeAPIServers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apifederation.APIServer, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(apiserversResource, name, data, subresources...), &apifederation.APIServer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apifederation.APIServer), err
}
