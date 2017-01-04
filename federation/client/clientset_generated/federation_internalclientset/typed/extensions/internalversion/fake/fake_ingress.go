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

// FakeIngresses implements IngressInterface
type FakeIngresses struct {
	Fake *FakeExtensions
	ns   string
}

var ingressesResource = unversioned.GroupVersionResource{Group: "extensions", Version: "", Resource: "ingresses"}

func (c *FakeIngresses) Create(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(ingressesResource, c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngresses) Update(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(ingressesResource, c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngresses) UpdateStatus(ingress *extensions.Ingress) (*extensions.Ingress, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(ingressesResource, "status", c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngresses) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(ingressesResource, c.ns, name), &extensions.Ingress{})

	return err
}

func (c *FakeIngresses) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(ingressesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.IngressList{})
	return err
}

func (c *FakeIngresses) Get(name string) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(ingressesResource, c.ns, name), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngresses) List(opts api.ListOptions) (result *extensions.IngressList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(ingressesResource, c.ns, opts), &extensions.IngressList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.IngressList{}
	for _, item := range obj.(*extensions.IngressList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested ingresses.
func (c *FakeIngresses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(ingressesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched ingress.
func (c *FakeIngresses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(ingressesResource, c.ns, name, data, subresources...), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}
