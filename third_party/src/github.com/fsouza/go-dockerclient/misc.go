// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
)

// Version returns version information about the docker server.
//
// See http://goo.gl/IqKNRE for more details.
func (c *Client) Version() (*Env, error) {
	body, _, err := c.do("GET", "/version", nil)
	if err != nil {
		return nil, err
	}
	var env Env
	if err := env.Decode(bytes.NewReader(body)); err != nil {
		return nil, err
	}
	return &env, nil
}

// Info returns system-wide information, like the number of running containers.
//
// See http://goo.gl/LOmySw for more details.
func (c *Client) Info() (*Env, error) {
	body, _, err := c.do("GET", "/info", nil)
	if err != nil {
		return nil, err
	}
	var info Env
	err = info.Decode(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return &info, nil
}
