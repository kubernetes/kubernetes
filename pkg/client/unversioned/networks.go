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

package unversioned

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type NetworksInterface interface {
	Networks() NetworkInterface
}

type NetworkInterface interface {
	Create(item *api.Network) (*api.Network, error)
	Get(name string) (result *api.Network, err error)
	List(label labels.Selector, field fields.Selector) (*api.NetworkList, error)
	Delete(name string) error
	Update(item *api.Network) (*api.Network, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	Status(item *api.Network) (*api.Network, error)
}

// networks implements NetworksInterface
type networks struct {
	r *Client
}

// newNetworks returns a networks object.
func newNetworks(c *Client) *networks {
	return &networks{r: c}
}

// Create creates a new network.
func (c *networks) Create(network *api.Network) (*api.Network, error) {
	result := &api.Network{}
	err := c.r.Post().Resource("networks").Body(network).Do().Into(result)
	return result, err
}

// List lists all the networks in the cluster.
func (c *networks) List(label labels.Selector, field fields.Selector) (*api.NetworkList, error) {
	result := &api.NetworkList{}
	err := c.r.Get().
	Resource("networks").
	LabelsSelectorParam(label).
	FieldsSelectorParam(field).
	Do().Into(result)
	return result, err
}

// Update takes the representation of a network to update.  Returns the server's representation of the network, and an error, if it occurs.
func (c *networks) Update(network *api.Network) (result *api.Network, err error) {
	result = &api.Network{}
	if len(network.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", network)
		return
	}
	err = c.r.Put().Resource("networks").Name(network.Name).Body(network).Do().Into(result)
	return
}

// Status takes the representation of a network to update.  Returns the server's representation of the network, and an error, if it occurs.
func (c *networks) Status(network *api.Network) (result *api.Network, err error) {
	result = &api.Network{}
	if len(network.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", network)
		return
	}
	err = c.r.Put().Resource("networks").Name(network.Name).SubResource("status").Body(network).Do().Into(result)
	return
}

// Get gets an existing network
func (c *networks) Get(name string) (*api.Network, error) {
	result := &api.Network{}
	err := c.r.Get().Resource("networks").Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing network.
func (c *networks) Delete(name string) error {
	return c.r.Delete().Resource("networks").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested networks.
func (c *networks) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
	Prefix("watch").
	Resource("networks").
	Param("resourceVersion", resourceVersion).
	LabelsSelectorParam(label).
	FieldsSelectorParam(field).
	Watch()
}
