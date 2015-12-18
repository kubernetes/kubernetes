/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// NodeNamespacer has methods to work with Node resources in a namespace
type NodeNamespacer interface {
	Nodes(namespace string) NodeInterface
}

// NodeInterface has methods to work with Node resources.
type NodeInterface interface {
	Create(*api.Node) (*api.Node, error)
	Update(*api.Node) (*api.Node, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.Node, error)
	List(opts unversioned.ListOptions) (*api.NodeList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// nodes implements NodeInterface
type nodes struct {
	client *LegacyClient
	ns     string
}

// newNodes returns a Nodes
func newNodes(c *LegacyClient, namespace string) *nodes {
	return &nodes{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a node and creates it.  Returns the server's representation of the node, and an error, if there is any.
func (c *nodes) Create(node *api.Node) (result *api.Node, err error) {
	result = &api.Node{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("nodes").
		Body(node).
		Do().
		Into(result)
	return
}

// Update takes the representation of a node and updates it. Returns the server's representation of the node, and an error, if there is any.
func (c *nodes) Update(node *api.Node) (result *api.Node, err error) {
	result = &api.Node{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("nodes").
		Name(node.Name).
		Body(node).
		Do().
		Into(result)
	return
}

// Delete takes name of the node and deletes it. Returns an error if one occurs.
func (c *nodes) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("nodes").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("nodes").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the node, and returns the corresponding node object, and an error if there is any.
func (c *nodes) Get(name string) (result *api.Node, err error) {
	result = &api.Node{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("nodes").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Nodes that match those selectors.
func (c *nodes) List(opts unversioned.ListOptions) (result *api.NodeList, err error) {
	result = &api.NodeList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("nodes").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested nodes.
func (c *nodes) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("nodes").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
