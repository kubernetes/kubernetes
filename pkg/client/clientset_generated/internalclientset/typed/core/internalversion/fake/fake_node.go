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
	api "k8s.io/kubernetes/pkg/api"
)

// FakeNodes implements NodeInterface
type FakeNodes struct {
	Fake *FakeCore
}

var nodesResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "nodes"}

var nodesKind = schema.GroupVersionKind{Group: "", Version: "", Kind: "Node"}

// Get takes name of the node, and returns the corresponding node object, and an error if there is any.
func (c *FakeNodes) Get(name string, options v1.GetOptions) (result *api.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(nodesResource, name), &api.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Node), err
}

// List takes label and field selectors, and returns the list of Nodes that match those selectors.
func (c *FakeNodes) List(opts v1.ListOptions) (result *api.NodeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(nodesResource, nodesKind, opts), &api.NodeList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.NodeList{}
	for _, item := range obj.(*api.NodeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested nodes.
func (c *FakeNodes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(nodesResource, opts))
}

// Create takes the representation of a node and creates it.  Returns the server's representation of the node, and an error, if there is any.
func (c *FakeNodes) Create(node *api.Node) (result *api.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(nodesResource, node), &api.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Node), err
}

// Update takes the representation of a node and updates it. Returns the server's representation of the node, and an error, if there is any.
func (c *FakeNodes) Update(node *api.Node) (result *api.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(nodesResource, node), &api.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Node), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeNodes) UpdateStatus(node *api.Node) (*api.Node, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(nodesResource, "status", node), &api.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Node), err
}

// Delete takes name of the node and deletes it. Returns an error if one occurs.
func (c *FakeNodes) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(nodesResource, name), &api.Node{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeNodes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(nodesResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.NodeList{})
	return err
}

// Patch applies the patch and returns the patched node.
func (c *FakeNodes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(nodesResource, name, data, subresources...), &api.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.Node), err
}
