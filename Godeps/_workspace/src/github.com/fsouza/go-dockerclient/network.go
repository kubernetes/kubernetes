// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

// ErrNetworkAlreadyExists is the error returned by CreateNetwork when the
// network already exists.
var ErrNetworkAlreadyExists = errors.New("network already exists")

// Network represents a network.
//
// See https://goo.gl/FDkCdQ for more details.
type Network struct {
	Name      string      `json:"name"`
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Endpoints []*Endpoint `json:"endpoints"`
}

// Endpoint represents an endpoint.
//
// See https://goo.gl/FDkCdQ for more details.
type Endpoint struct {
	Name    string `json:"name"`
	ID      string `json:"id"`
	Network string `json:"network"`
}

// ListNetworks returns all networks.
//
// See https://goo.gl/4hCNtZ for more details.
func (c *Client) ListNetworks() ([]Network, error) {
	body, _, err := c.do("GET", "/networks", doOptions{})
	if err != nil {
		return nil, err
	}
	var networks []Network
	if err := json.Unmarshal(body, &networks); err != nil {
		return nil, err
	}
	return networks, nil
}

// NetworkInfo returns information about a network by its ID.
//
// See https://goo.gl/4hCNtZ for more details.
func (c *Client) NetworkInfo(id string) (*Network, error) {
	path := "/networks/" + id
	body, status, err := c.do("GET", path, doOptions{})
	if status == http.StatusNotFound {
		return nil, &NoSuchNetwork{ID: id}
	}
	if err != nil {
		return nil, err
	}
	var network Network
	if err := json.Unmarshal(body, &network); err != nil {
		return nil, err
	}
	return &network, nil
}

// CreateNetworkOptions specify parameters to the CreateNetwork function and
// (for now) is the expected body of the "create network" http request message
//
// See https://goo.gl/FDkCdQ for more details.
type CreateNetworkOptions struct {
	Name        string                 `json:"name"`
	NetworkType string                 `json:"network_type"`
	Options     map[string]interface{} `json:"options"`
}

// CreateNetwork creates a new network, returning the network instance,
// or an error in case of failure.
//
// See http://goo.gl/mErxNp for more details.
func (c *Client) CreateNetwork(opts CreateNetworkOptions) (*Network, error) {
	body, status, err := c.do(
		"POST",
		"/networks",
		doOptions{
			data: opts,
		},
	)

	if status == http.StatusConflict {
		return nil, ErrNetworkAlreadyExists
	}
	if err != nil {
		return nil, err
	}

	type createNetworkResponse struct {
		ID string
	}
	var (
		network Network
		resp    createNetworkResponse
	)
	err = json.Unmarshal(body, &resp)
	if err != nil {
		return nil, err
	}

	network.Name = opts.Name
	network.ID = resp.ID
	network.Type = opts.NetworkType

	return &network, nil
}

// NoSuchNetwork is the error returned when a given network does not exist.
type NoSuchNetwork struct {
	ID string
}

func (err *NoSuchNetwork) Error() string {
	return fmt.Sprintf("No such network: %s", err.ID)
}
