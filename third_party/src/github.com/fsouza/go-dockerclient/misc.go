// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"io"

	"github.com/fsouza/go-dockerclient/engine"
)

// Version returns version information about the docker server.
//
// See http://goo.gl/IqKNRE for more details.
func (c *Client) Version() (*engine.Env, error) {
	body, _, err := c.do("GET", "/version", nil)
	if err != nil {
		return nil, err
	}
	out := engine.NewOutput()
	remoteVersion, err := out.AddEnv()
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(out, bytes.NewReader(body)); err != nil {
		return nil, err
	}
	return remoteVersion, nil
}

// Info returns system-wide information, like the number of running containers.
//
// See http://goo.gl/LOmySw for more details.
func (c *Client) Info() (*engine.Env, error) {
	body, _, err := c.do("GET", "/info", nil)
	if err != nil {
		return nil, err
	}
	var info engine.Env
	err = info.Decode(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return &info, nil
}
