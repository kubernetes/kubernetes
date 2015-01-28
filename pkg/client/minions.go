/*
Copyright 2014 Google Inc. All rights reserved.

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
	"errors"
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type NodesInterface interface {
	Nodes() NodeInterface
}

type NodeInterface interface {
	Get(name string) (result *api.Node, err error)
	Create(minion *api.Node) (*api.Node, error)
	List() (*api.NodeList, error)
	Delete(name string) error
	Update(*api.Node) (*api.Node, error)
}

// nodes implements NodesInterface
type nodes struct {
	r *Client
}

// newNodes returns a nodes object. Uses "minions" as the
// URL resource name for v1beta1 and v1beta2.
func newNodes(c *Client) *nodes {
	return &nodes{c}
}

// resourceName returns node's URL resource name based on resource version.
func (c *nodes) resourceName() string {
	if preV1Beta3(c.r.APIVersion()) {
		return "minions"
	}
	return "nodes"
}

// Create creates a new node.
func (c *nodes) Create(minion *api.Node) (*api.Node, error) {
	result := &api.Node{}
	err := c.r.Post().Resource(c.resourceName()).Body(minion).Do().Into(result)
	return result, err
}

// List lists all the nodes in the cluster.
func (c *nodes) List() (*api.NodeList, error) {
	result := &api.NodeList{}
	err := c.r.Get().Resource(c.resourceName()).Do().Into(result)
	return result, err
}

// Get gets an existing node.
func (c *nodes) Get(name string) (*api.Node, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.Node{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing node.
func (c *nodes) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

// Update updates an existing node.
func (c *nodes) Update(minion *api.Node) (*api.Node, error) {
	result := &api.Node{}
	if len(minion.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", minion)
		return nil, err
	}
	err := c.r.Put().Resource(c.resourceName()).Name(minion.Name).Body(minion).Do().Into(result)
	return result, err
}
