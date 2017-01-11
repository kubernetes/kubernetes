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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeEndpoints implements EndpointsInterface
type FakeEndpoints struct {
	Fake *FakeCoreV1
	ns   string
}

var endpointsResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "endpoints"}

func (c *FakeEndpoints) Create(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(endpointsResource, c.ns, endpoints), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) Update(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(endpointsResource, c.ns, endpoints), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(endpointsResource, c.ns, name), &v1.Endpoints{})

	return err
}

func (c *FakeEndpoints) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(endpointsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.EndpointsList{})
	return err
}

func (c *FakeEndpoints) Get(name string, options meta_v1.GetOptions) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(endpointsResource, c.ns, name), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) List(opts v1.ListOptions) (result *v1.EndpointsList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(endpointsResource, c.ns, opts), &v1.EndpointsList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.EndpointsList{}
	for _, item := range obj.(*v1.EndpointsList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested endpoints.
func (c *FakeEndpoints) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(endpointsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched endpoints.
func (c *FakeEndpoints) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(endpointsResource, c.ns, name, data, subresources...), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}
