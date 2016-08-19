//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package client

import (
	"bytes"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"net/http"
)

func (c *Client) ClusterCreate() (*api.ClusterInfoResponse, error) {

	// Create a request
	req, err := http.NewRequest("POST", c.host+"/clusters", bytes.NewBuffer([]byte(`{}`)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	// Set token
	err = c.setToken(req)
	if err != nil {
		return nil, err
	}

	// Send request
	r, err := c.do(req)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusCreated {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var cluster api.ClusterInfoResponse
	err = utils.GetJsonFromResponse(r, &cluster)
	r.Body.Close()
	if err != nil {
		return nil, err
	}

	return &cluster, nil
}

func (c *Client) ClusterInfo(id string) (*api.ClusterInfoResponse, error) {

	// Create request
	req, err := http.NewRequest("GET", c.host+"/clusters/"+id, nil)
	if err != nil {
		return nil, err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return nil, err
	}

	// Get info
	r, err := c.do(req)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var cluster api.ClusterInfoResponse
	err = utils.GetJsonFromResponse(r, &cluster)
	r.Body.Close()
	if err != nil {
		return nil, err
	}

	return &cluster, nil
}

func (c *Client) ClusterList() (*api.ClusterListResponse, error) {

	// Create request
	req, err := http.NewRequest("GET", c.host+"/clusters", nil)
	if err != nil {
		return nil, err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return nil, err
	}

	// Get info
	r, err := c.do(req)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var clusters api.ClusterListResponse
	err = utils.GetJsonFromResponse(r, &clusters)
	if err != nil {
		return nil, err
	}

	return &clusters, nil
}

func (c *Client) ClusterDelete(id string) error {

	// Create DELETE request
	req, err := http.NewRequest("DELETE", c.host+"/clusters/"+id, nil)
	if err != nil {
		return err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return err
	}

	// Send request
	r, err := c.do(req)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusOK {
		return utils.GetErrorFromResponse(r)
	}

	return nil
}
