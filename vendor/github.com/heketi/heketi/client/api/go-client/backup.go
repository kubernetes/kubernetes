//
// Copyright (c) 2016 The heketi Authors
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
	"io"
	"net/http"

	"github.com/heketi/heketi/pkg/utils"
)

func (c *Client) BackupDb(w io.Writer) error {
	// Create a request
	req, err := http.NewRequest("GET", c.host+"/backup/db", nil)
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

	// Read data from response
	defer r.Body.Close()
	_, err = io.Copy(w, r.Body)

	return err
}
