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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type MinionsInterface interface {
	Minions() MinionInterface
}

type MinionInterface interface {
	Get(id string) (result *api.Minion, err error)
	Create(minion *api.Minion) (*api.Minion, error)
	List() (*api.MinionList, error)
	Delete(id string) error
}

// MinionsClient implements Minions interface
type MinionsClient struct {
	r *Client
}

// NewMinionsClient returns a MinionsClient
func NewMinionsClient(c *Client) *MinionsClient {
	return &MinionsClient{c}
}

// Create creates a new minion.
func (c *MinionsClient) Create(minion *api.Minion) (*api.Minion, error) {
	result := &api.Minion{}
	err := c.r.Post().Path("minions").Body(minion).Do().Into(result)
	return result, err
}

// List lists all the minions in the cluster.
func (c *MinionsClient) List() (result *api.MinionList, err error) {
	result = &api.MinionList{}
	err = c.r.Get().Path("minions").Do().Into(result)
	return
}

// Get gets an existing minion
func (c *MinionsClient) Get(id string) (result *api.Minion, err error) {
	result = &api.Minion{}
	err = c.r.Get().Path("minions").Path(id).Do().Into(result)
	return
}

// Delete deletes an existing minion.
func (c *MinionsClient) Delete(id string) error {
	return c.r.Delete().Path("minions").Path(id).Do().Error()
}
