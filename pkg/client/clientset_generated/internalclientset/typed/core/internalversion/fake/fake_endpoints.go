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
	api "k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeEndpoints implements EndpointsInterface
type FakeEndpoints struct {
	Fake *FakeCore
	ns   string
}

var endpointsResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "endpoints"}

func (c *FakeEndpoints) Create(endpoints *api.Endpoints) (result *api.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(endpointsResource, c.ns, endpoints), &api.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) Update(endpoints *api.Endpoints) (result *api.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(endpointsResource, c.ns, endpoints), &api.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(endpointsResource, c.ns, name), &api.Endpoints{})

	return err
}

func (c *FakeEndpoints) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(endpointsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.EndpointsList{})
	return err
}

func (c *FakeEndpoints) Get(name string) (result *api.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(endpointsResource, c.ns, name), &api.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Endpoints), err
}

func (c *FakeEndpoints) List(opts api.ListOptions) (result *api.EndpointsList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(endpointsResource, c.ns, opts), &api.EndpointsList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.EndpointsList{}
	for _, item := range obj.(*api.EndpointsList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested endpoints.
func (c *FakeEndpoints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(endpointsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched endpoints.
func (c *FakeEndpoints) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(endpointsResource, c.ns, name, data, subresources...), &api.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Endpoints), err
}
