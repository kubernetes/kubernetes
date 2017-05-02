//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
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
