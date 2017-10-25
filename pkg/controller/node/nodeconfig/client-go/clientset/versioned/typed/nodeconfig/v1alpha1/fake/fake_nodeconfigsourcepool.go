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
	v1alpha1 "k8s.io/kubernetes/pkg/controller/node/nodeconfig/v1alpha1"
)

// FakeNodeConfigSourcePools implements NodeConfigSourcePoolInterface
type FakeNodeConfigSourcePools struct {
	Fake *FakeNodeconfigV1alpha1
}

var nodeconfigsourcepoolsResource = schema.GroupVersionResource{Group: "nodeconfig", Version: "v1alpha1", Resource: "nodeconfigsourcepools"}

var nodeconfigsourcepoolsKind = schema.GroupVersionKind{Group: "nodeconfig", Version: "v1alpha1", Kind: "NodeConfigSourcePool"}

// Get takes name of the nodeConfigSourcePool, and returns the corresponding nodeConfigSourcePool object, and an error if there is any.
func (c *FakeNodeConfigSourcePools) Get(name string, options v1.GetOptions) (result *v1alpha1.NodeConfigSourcePool, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(nodeconfigsourcepoolsResource, name), &v1alpha1.NodeConfigSourcePool{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.NodeConfigSourcePool), err
}

// List takes label and field selectors, and returns the list of NodeConfigSourcePools that match those selectors.
func (c *FakeNodeConfigSourcePools) List(opts v1.ListOptions) (result *v1alpha1.NodeConfigSourcePoolList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(nodeconfigsourcepoolsResource, nodeconfigsourcepoolsKind, opts), &v1alpha1.NodeConfigSourcePoolList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.NodeConfigSourcePoolList{}
	for _, item := range obj.(*v1alpha1.NodeConfigSourcePoolList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested nodeConfigSourcePools.
func (c *FakeNodeConfigSourcePools) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(nodeconfigsourcepoolsResource, opts))
}

// Create takes the representation of a nodeConfigSourcePool and creates it.  Returns the server's representation of the nodeConfigSourcePool, and an error, if there is any.
func (c *FakeNodeConfigSourcePools) Create(nodeConfigSourcePool *v1alpha1.NodeConfigSourcePool) (result *v1alpha1.NodeConfigSourcePool, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(nodeconfigsourcepoolsResource, nodeConfigSourcePool), &v1alpha1.NodeConfigSourcePool{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.NodeConfigSourcePool), err
}

// Update takes the representation of a nodeConfigSourcePool and updates it. Returns the server's representation of the nodeConfigSourcePool, and an error, if there is any.
func (c *FakeNodeConfigSourcePools) Update(nodeConfigSourcePool *v1alpha1.NodeConfigSourcePool) (result *v1alpha1.NodeConfigSourcePool, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(nodeconfigsourcepoolsResource, nodeConfigSourcePool), &v1alpha1.NodeConfigSourcePool{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.NodeConfigSourcePool), err
}

// Delete takes name of the nodeConfigSourcePool and deletes it. Returns an error if one occurs.
func (c *FakeNodeConfigSourcePools) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(nodeconfigsourcepoolsResource, name), &v1alpha1.NodeConfigSourcePool{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeNodeConfigSourcePools) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(nodeconfigsourcepoolsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.NodeConfigSourcePoolList{})
	return err
}

// Patch applies the patch and returns the patched nodeConfigSourcePool.
func (c *FakeNodeConfigSourcePools) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.NodeConfigSourcePool, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(nodeconfigsourcepoolsResource, name, data, subresources...), &v1alpha1.NodeConfigSourcePool{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.NodeConfigSourcePool), err
}
