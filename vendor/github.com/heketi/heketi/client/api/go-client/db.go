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
	"io/ioutil"
	"net/http"

	"github.com/heketi/heketi/pkg/utils"
)

// DbDump provides a JSON representation of current state of DB
func (c *Client) DbDump() (string, error) {
	req, err := http.NewRequest("GET", c.host+"/db/dump", nil)
	if err != nil {
		return "", err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return "", err
	}

	// Send request
	r, err := c.do(req)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return "", utils.GetErrorFromResponse(r)
	}

	respBytes, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return "", err
	}

	respJSON := string(respBytes)
	return respJSON, nil
}

// DbCheck provides a JSON summary of the DB check operation
func (c *Client) DbCheck() (string, error) {
	req, err := http.NewRequest("GET", c.host+"/db/check", nil)
	if err != nil {
		return "", err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return "", err
	}

	// Send request
	r, err := c.do(req)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return "", utils.GetErrorFromResponse(r)
	}

	respBytes, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return "", err
	}

	respJSON := string(respBytes)
	return respJSON, nil
}
