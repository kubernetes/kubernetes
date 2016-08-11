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
	api "k8s.io/client-go/1.4/pkg/api"
	unversioned "k8s.io/client-go/1.4/pkg/api/unversioned"
	v1 "k8s.io/client-go/1.4/pkg/api/v1"
	labels "k8s.io/client-go/1.4/pkg/labels"
	watch "k8s.io/client-go/1.4/pkg/watch"
	testing "k8s.io/client-go/1.4/testing"
)

// FakeEndpoints implements EndpointsInterface
type FakeEndpoints struct {
	Fake *FakeCore
	ns   string
}

var endpointsResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "endpoints"}

func (c *FakeEndpoints) Create(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(endpointsResource, c.ns, endpoints), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) Update(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(endpointsResource, c.ns, endpoints), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(endpointsResource, c.ns, name), &v1.Endpoints{})

	return err
}

func (c *FakeEndpoints) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(endpointsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.EndpointsList{})
	return err
}

func (c *FakeEndpoints) Get(name string) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(endpointsResource, c.ns, name), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}

func (c *FakeEndpoints) List(opts api.ListOptions) (result *v1.EndpointsList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(endpointsResource, c.ns, opts), &v1.EndpointsList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
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
func (c *FakeEndpoints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(endpointsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched endpoints.
func (c *FakeEndpoints) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Endpoints, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(endpointsResource, c.ns, name, data, subresources...), &v1.Endpoints{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Endpoints), err
}
