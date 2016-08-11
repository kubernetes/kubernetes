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
	v1beta1 "k8s.io/client-go/1.4/pkg/apis/extensions/v1beta1"
	labels "k8s.io/client-go/1.4/pkg/labels"
	watch "k8s.io/client-go/1.4/pkg/watch"
	testing "k8s.io/client-go/1.4/testing"
)

// FakeIngresses implements IngressInterface
type FakeIngresses struct {
	Fake *FakeExtensions
	ns   string
}

var ingressesResource = unversioned.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "ingresses"}

func (c *FakeIngresses) Create(ingress *v1beta1.Ingress) (result *v1beta1.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(ingressesResource, c.ns, ingress), &v1beta1.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Ingress), err
}

func (c *FakeIngresses) Update(ingress *v1beta1.Ingress) (result *v1beta1.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(ingressesResource, c.ns, ingress), &v1beta1.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Ingress), err
}

func (c *FakeIngresses) UpdateStatus(ingress *v1beta1.Ingress) (*v1beta1.Ingress, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(ingressesResource, "status", c.ns, ingress), &v1beta1.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Ingress), err
}

func (c *FakeIngresses) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(ingressesResource, c.ns, name), &v1beta1.Ingress{})

	return err
}

func (c *FakeIngresses) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(ingressesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.IngressList{})
	return err
}

func (c *FakeIngresses) Get(name string) (result *v1beta1.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(ingressesResource, c.ns, name), &v1beta1.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Ingress), err
}

func (c *FakeIngresses) List(opts api.ListOptions) (result *v1beta1.IngressList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(ingressesResource, c.ns, opts), &v1beta1.IngressList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.IngressList{}
	for _, item := range obj.(*v1beta1.IngressList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested ingresses.
func (c *FakeIngresses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(ingressesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched ingress.
func (c *FakeIngresses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(ingressesResource, c.ns, name, data, subresources...), &v1beta1.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Ingress), err
}
