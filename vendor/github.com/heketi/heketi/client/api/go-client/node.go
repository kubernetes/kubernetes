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
	"encoding/json"
	"net/http"

	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
)

func (c *Client) NodeAdd(request *api.NodeAddRequest) (*api.NodeInfoResponse, error) {

	// Marshal request to JSON
	buffer, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	// Create a request
	req, err := http.NewRequest("POST", c.host+"/nodes", bytes.NewBuffer(buffer))
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
	defer r.Body.Close()
	if r.StatusCode != http.StatusAccepted {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.pollResponse(r)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var node api.NodeInfoResponse
	err = utils.GetJsonFromResponse(r, &node)
	if err != nil {
		return nil, err
	}

	return &node, nil
}

func (c *Client) NodeInfo(id string) (*api.NodeInfoResponse, error) {

	// Create request
	req, err := http.NewRequest("GET", c.host+"/nodes/"+id, nil)
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
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var node api.NodeInfoResponse
	err = utils.GetJsonFromResponse(r, &node)
	if err != nil {
		return nil, err
	}

	return &node, nil
}

func (c *Client) NodeDelete(id string) error {

	// Create a request
	req, err := http.NewRequest("DELETE", c.host+"/nodes/"+id, nil)
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
	defer r.Body.Close()
	if r.StatusCode != http.StatusAccepted {
		return utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.pollResponse(r)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusNoContent {
		return utils.GetErrorFromResponse(r)
	}

	return nil
}

func (c *Client) NodeState(id string, request *api.StateRequest) error {
	// Marshal request to JSON
	buffer, err := json.Marshal(request)
	if err != nil {
		return err
	}

	// Create a request
	req, err := http.NewRequest("POST",
		c.host+"/nodes/"+id+"/state",
		bytes.NewBuffer(buffer))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	// Set token
	err = c.setToken(req)
	if err != nil {
		return err
	}

	// Get info
	r, err := c.do(req)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusAccepted {
		return utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.pollResponse(r)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusNoContent {
		return utils.GetErrorFromResponse(r)
	}

	return nil
}

func (c *Client) NodeSetTags(id string, request *api.TagsChangeRequest) error {
	buffer, err := json.Marshal(request)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST",
		c.host+"/nodes/"+id+"/tags",
		bytes.NewBuffer(buffer))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	// Set token
	err = c.setToken(req)
	if err != nil {
		return err
	}

	// Get info
	r, err := c.do(req)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return utils.GetErrorFromResponse(r)
	}
	return nil
}
