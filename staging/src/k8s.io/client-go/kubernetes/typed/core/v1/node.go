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

package v1

import (
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/client-go/pkg/api"
	v1 "k8s.io/client-go/pkg/api/v1"
	rest "k8s.io/client-go/rest"
)

// NodesGetter has a method to return a NodeInterface.
// A group's client should implement this interface.
type NodesGetter interface {
	Nodes() NodeInterface
}

// NodeInterface has methods to work with Node resources.
type NodeInterface interface {
	Create(*v1.Node) (*v1.Node, error)
	Update(*v1.Node) (*v1.Node, error)
	UpdateStatus(*v1.Node) (*v1.Node, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options meta_v1.GetOptions) (*v1.Node, error)
	List(opts v1.ListOptions) (*v1.NodeList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Node, err error)
	NodeExpansion
}

// nodes implements NodeInterface
type nodes struct {
	client rest.Interface
}

// newNodes returns a Nodes
func newNodes(c *CoreV1Client) *nodes {
	return &nodes{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a node and creates it.  Returns the server's representation of the node, and an error, if there is any.
func (c *nodes) Create(node *v1.Node) (result *v1.Node, err error) {
	result = &v1.Node{}
	err = c.client.Post().
		Resource("nodes").
		Body(node).
		Do().
		Into(result)
	return
}

// Update takes the representation of a node and updates it. Returns the server's representation of the node, and an error, if there is any.
func (c *nodes) Update(node *v1.Node) (result *v1.Node, err error) {
	result = &v1.Node{}
	err = c.client.Put().
		Resource("nodes").
		Name(node.Name).
		Body(node).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *nodes) UpdateStatus(node *v1.Node) (result *v1.Node, err error) {
	result = &v1.Node{}
	err = c.client.Put().
		Resource("nodes").
		Name(node.Name).
		SubResource("status").
		Body(node).
		Do().
		Into(result)
	return
}

// Delete takes name of the node and deletes it. Returns an error if one occurs.
func (c *nodes) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("nodes").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *nodes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("nodes").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the node, and returns the corresponding node object, and an error if there is any.
func (c *nodes) Get(name string, options meta_v1.GetOptions) (result *v1.Node, err error) {
	result = &v1.Node{}
	err = c.client.Get().
		Resource("nodes").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Nodes that match those selectors.
func (c *nodes) List(opts v1.ListOptions) (result *v1.NodeList, err error) {
	result = &v1.NodeList{}
	err = c.client.Get().
		Resource("nodes").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested nodes.
func (c *nodes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("nodes").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched node.
func (c *nodes) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Node, err error) {
	result = &v1.Node{}
	err = c.client.Patch(pt).
		Resource("nodes").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
