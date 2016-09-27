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
	api "k8s.io/client-go/1.5/pkg/api"
	unversioned "k8s.io/client-go/1.5/pkg/api/unversioned"
	v1 "k8s.io/client-go/1.5/pkg/api/v1"
	labels "k8s.io/client-go/1.5/pkg/labels"
	watch "k8s.io/client-go/1.5/pkg/watch"
	testing "k8s.io/client-go/1.5/testing"
)

// FakeServices implements ServiceInterface
type FakeServices struct {
	Fake *FakeCore
	ns   string
}

var servicesResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "services"}

func (c *FakeServices) Create(service *v1.Service) (result *v1.Service, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(servicesResource, c.ns, service), &v1.Service{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Service), err
}

func (c *FakeServices) Update(service *v1.Service) (result *v1.Service, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(servicesResource, c.ns, service), &v1.Service{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Service), err
}

func (c *FakeServices) UpdateStatus(service *v1.Service) (*v1.Service, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(servicesResource, "status", c.ns, service), &v1.Service{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Service), err
}

func (c *FakeServices) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(servicesResource, c.ns, name), &v1.Service{})

	return err
}

func (c *FakeServices) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(servicesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.ServiceList{})
	return err
}

func (c *FakeServices) Get(name string) (result *v1.Service, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(servicesResource, c.ns, name), &v1.Service{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Service), err
}

func (c *FakeServices) List(opts api.ListOptions) (result *v1.ServiceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(servicesResource, c.ns, opts), &v1.ServiceList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.ServiceList{}
	for _, item := range obj.(*v1.ServiceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested services.
func (c *FakeServices) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(servicesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched service.
func (c *FakeServices) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Service, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(servicesResource, c.ns, name, data, subresources...), &v1.Service{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Service), err
}
