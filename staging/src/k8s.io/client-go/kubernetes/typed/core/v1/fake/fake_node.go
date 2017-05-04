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
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	v1 "k8s.io/client-go/pkg/api/v1"
	testing "k8s.io/client-go/testing"
)

// FakeNodes implements NodeInterface
type FakeNodes struct {
	Fake *FakeCoreV1
}

var nodesResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "nodes"}

var nodesKind = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Node"}

func (c *FakeNodes) Create(node *v1.Node) (result *v1.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(nodesResource, node), &v1.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Node), err
}

func (c *FakeNodes) Update(node *v1.Node) (result *v1.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(nodesResource, node), &v1.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Node), err
}

func (c *FakeNodes) UpdateStatus(node *v1.Node) (*v1.Node, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(nodesResource, "status", node), &v1.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Node), err
}

func (c *FakeNodes) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(nodesResource, name), &v1.Node{})
	return err
}

func (c *FakeNodes) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(nodesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1.NodeList{})
	return err
}

func (c *FakeNodes) Get(name string, options meta_v1.GetOptions) (result *v1.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(nodesResource, name), &v1.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Node), err
}

func (c *FakeNodes) List(opts meta_v1.ListOptions) (result *v1.NodeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(nodesResource, nodesKind, opts), &v1.NodeList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.NodeList{}
	for _, item := range obj.(*v1.NodeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested nodes.
func (c *FakeNodes) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(nodesResource, opts))
}

// Patch applies the patch and returns the patched node.
func (c *FakeNodes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.Node, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(nodesResource, name, data, subresources...), &v1.Node{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Node), err
}
