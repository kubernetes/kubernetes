// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"strings"
)

// Version returns version information about the docker server.
//
// See http://goo.gl/BOZrF5 for more details.
func (c *Client) Version() (*Env, error) {
	body, _, err := c.do("GET", "/version", doOptions{})
	if err != nil {
		return nil, err
	}
	var env Env
	if err := env.Decode(bytes.NewReader(body)); err != nil {
		return nil, err
	}
	return &env, nil
}

// Info returns system-wide information about the Docker server.
//
// See http://goo.gl/wmqZsW for more details.
func (c *Client) Info() (*Env, error) {
	body, _, err := c.do("GET", "/info", doOptions{})
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

// ParseRepositoryTag gets the name of the repository and returns it splitted
// in two parts: the repository and the tag.
//
// Some examples:
//
//     localhost.localdomain:5000/samalba/hipache:latest -> localhost.localdomain:5000/samalba/hipache, latest
//     localhost.localdomain:5000/samalba/hipache -> localhost.localdomain:5000/samalba/hipache, ""
func ParseRepositoryTag(repoTag string) (repository string, tag string) {
	n := strings.LastIndex(repoTag, ":")
	if n < 0 {
		return repoTag, ""
	}
	if tag := repoTag[n+1:]; !strings.Contains(tag, "/") {
		return repoTag[:n], tag
	}
	return repoTag, ""
}
