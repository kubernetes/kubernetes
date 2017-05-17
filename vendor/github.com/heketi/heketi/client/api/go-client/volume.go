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
	"time"

	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
)

func (c *Client) VolumeCreate(request *api.VolumeCreateRequest) (
	*api.VolumeInfoResponse, error) {

	// Marshal request to JSON
	buffer, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	// Create a request
	req, err := http.NewRequest("POST",
		c.host+"/volumes",
		bytes.NewBuffer(buffer))
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
	if r.StatusCode != http.StatusAccepted {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.waitForResponseWithTimer(r, time.Second)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var volume api.VolumeInfoResponse
	err = utils.GetJsonFromResponse(r, &volume)
	r.Body.Close()
	if err != nil {
		return nil, err
	}

	return &volume, nil

}

func (c *Client) VolumeExpand(id string, request *api.VolumeExpandRequest) (
	*api.VolumeInfoResponse, error) {

	// Marshal request to JSON
	buffer, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	// Create a request
	req, err := http.NewRequest("POST",
		c.host+"/volumes/"+id+"/expand",
		bytes.NewBuffer(buffer))
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
	if r.StatusCode != http.StatusAccepted {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.waitForResponseWithTimer(r, time.Second)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}

	// Read JSON response
	var volume api.VolumeInfoResponse
	err = utils.GetJsonFromResponse(r, &volume)
	r.Body.Close()
	if err != nil {
		return nil, err
	}

	return &volume, nil

}

func (c *Client) VolumeList() (*api.VolumeListResponse, error) {

	// Create request
	req, err := http.NewRequest("GET", c.host+"/volumes", nil)
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
	var volumes api.VolumeListResponse
	err = utils.GetJsonFromResponse(r, &volumes)
	if err != nil {
		return nil, err
	}

	return &volumes, nil
}

func (c *Client) VolumeInfo(id string) (*api.VolumeInfoResponse, error) {

	// Create request
	req, err := http.NewRequest("GET", c.host+"/volumes/"+id, nil)
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
	var volume api.VolumeInfoResponse
	err = utils.GetJsonFromResponse(r, &volume)
	r.Body.Close()
	if err != nil {
		return nil, err
	}

	return &volume, nil
}

func (c *Client) VolumeDelete(id string) error {

	// Create a request
	req, err := http.NewRequest("DELETE", c.host+"/volumes/"+id, nil)
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
	if r.StatusCode != http.StatusAccepted {
		return utils.GetErrorFromResponse(r)
	}

	// Wait for response
	r, err = c.waitForResponseWithTimer(r, time.Second)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusNoContent {
		return utils.GetErrorFromResponse(r)
	}

	return nil
}
