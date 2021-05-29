//
// Copyright (c) 2019 The heketi Authors
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
	"io"
	"net/http"

	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
)

// BrickEvict requests that Heketi evict the given brick from the
// underlying gluster volume, automatically replacing it with a new brick.
//
// NOTE: options is currently empty but reserved for future extensions
// to the api.
func (c *Client) BrickEvict(id string, request *api.BrickEvictOptions) error {
	var buf io.Reader
	if request != nil {
		b, err := json.Marshal(request)
		if err != nil {
			return err
		}
		buf = bytes.NewBuffer(b)
	}

	// Create a request
	req, err := http.NewRequest("POST", c.host+"/bricks/to-evict/"+id, buf)
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
