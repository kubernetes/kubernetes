/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type NodesInterface interface {
	Nodes() NodeInterface
}

type NodeInterface interface {
	Get(name string) (result *api.Node, err error)
	Create(node *api.Node) (*api.Node, error)
	List(label labels.Selector, field fields.Selector) (*api.NodeList, error)
	Delete(name string) error
	Update(*api.Node) (*api.Node, error)
	UpdateStatus(*api.Node) (*api.Node, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// nodes implements NodesInterface
type nodes struct {
	r *Client
}

// newNodes returns a nodes object.
func newNodes(c *Client) *nodes {
	return &nodes{c}
}

// resourceName returns node's URL resource name.
func (c *nodes) resourceName() string {
	return "nodes"
}

// Create creates a new node.
func (c *nodes) Create(node *api.Node) (*api.Node, error) {
	result := &api.Node{}
	err := c.r.Post().Resource(c.resourceName()).Body(node).Do().Into(result)
	return result, err
}

// List takes a selector, and returns the list of nodes that match that selector in the cluster.
func (c *nodes) List(label labels.Selector, field fields.Selector) (*api.NodeList, error) {
	result := &api.NodeList{}
	err := c.r.Get().Resource(c.resourceName()).LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return result, err
}

// Get gets an existing node.
func (c *nodes) Get(name string) (*api.Node, error) {
	result := &api.Node{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing node.
func (c *nodes) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

// Update updates an existing node.
func (c *nodes) Update(node *api.Node) (*api.Node, error) {
	result := &api.Node{}
	if len(node.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", node)
		return nil, err
	}
	err := c.r.Put().Resource(c.resourceName()).Name(node.Name).Body(node).Do().Into(result)
	return result, err
}

func (c *nodes) UpdateStatus(node *api.Node) (*api.Node, error) {
	result := &api.Node{}
	if len(node.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", node)
		return nil, err
	}
	err := c.r.Put().Resource(c.resourceName()).Name(node.Name).SubResource("status").Body(node).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested nodes.
func (c *nodes) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(api.NamespaceAll).
		Resource(c.resourceName()).
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
