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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// FakeNetworks implements NetworkInterface
type FakeNetworks struct {
	Fake *FakeExtensionsV1beta1
	ns   string
}

var networksResource = schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "networks"}

var networksKind = schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Network"}

func (c *FakeNetworks) Create(network *v1beta1.Network) (result *v1beta1.Network, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(networksResource, c.ns, network), &v1beta1.Network{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Network), err
}

func (c *FakeNetworks) Update(network *v1beta1.Network) (result *v1beta1.Network, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(networksResource, c.ns, network), &v1beta1.Network{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Network), err
}

func (c *FakeNetworks) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(networksResource, c.ns, name), &v1beta1.Network{})

	return err
}

func (c *FakeNetworks) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(networksResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.NetworkList{})
	return err
}

func (c *FakeNetworks) Get(name string, options v1.GetOptions) (result *v1beta1.Network, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(networksResource, c.ns, name), &v1beta1.Network{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Network), err
}

func (c *FakeNetworks) List(opts v1.ListOptions) (result *v1beta1.NetworkList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(networksResource, networksKind, c.ns, opts), &v1beta1.NetworkList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.NetworkList{}
	for _, item := range obj.(*v1beta1.NetworkList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested networks.
func (c *FakeNetworks) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(networksResource, c.ns, opts))

}

// Patch applies the patch and returns the patched network.
func (c *FakeNetworks) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Network, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(networksResource, c.ns, name, data, subresources...), &v1beta1.Network{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Network), err
}
