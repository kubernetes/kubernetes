//
// Copyright (c) 2018 The heketi Authors
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

func (c *Client) OperationsInfo() (*api.OperationsInfo, error) {
	req, err := http.NewRequest("GET", c.host+"/operations", nil)
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
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}
	var oi api.OperationsInfo
	err = utils.GetJsonFromResponse(r, &oi)
	if err != nil {
		return nil, err
	}
	return &oi, nil
}

func (c *Client) PendingOperationList() (*api.PendingOperationListResponse, error) {
	req, err := http.NewRequest("GET", c.host+"/operations/pending", nil)
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
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}
	var ol api.PendingOperationListResponse
	err = utils.GetJsonFromResponse(r, &ol)
	if err != nil {
		return nil, err
	}
	return &ol, nil
}

func (c *Client) PendingOperationDetails(
	id string) (*api.PendingOperationDetails, error) {

	req, err := http.NewRequest("GET", c.host+"/operations/pending/"+id, nil)
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
	if r.StatusCode != http.StatusOK {
		return nil, utils.GetErrorFromResponse(r)
	}
	var pd api.PendingOperationDetails
	err = utils.GetJsonFromResponse(r, &pd)
	if err != nil {
		return nil, err
	}
	return &pd, nil
}

func (c *Client) PendingOperationCleanUp(
	request *api.PendingOperationsCleanRequest) error {

	buffer, err := json.Marshal(request)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST",
		c.host+"/operations/pending/cleanup",
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

	r, err := c.do(req)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	switch r.StatusCode {
	case http.StatusAccepted: // expected
	case http.StatusOK:
		return nil
	default:
		return utils.GetErrorFromResponse(r)
	}

	// NOTE: I (jjm) wanted this to truly async at first. But in
	// order to not deviate from the current model too much
	// AND that the rest async framework in heketi needs to be
	// polled in order to remove things from its map, the traditional
	// poll server after request behavior is retained here.
	r, err = c.pollResponse(r)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusNoContent {
		return utils.GetErrorFromResponse(r)
	}
	return nil
}
