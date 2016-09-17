// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"errors"
	"net/http"

	"golang.org/x/net/context"
)

var (
	// ErrNoSuchVolume is the error returned when the volume does not exist.
	ErrNoSuchVolume = errors.New("no such volume")

	// ErrVolumeInUse is the error returned when the volume requested to be removed is still in use.
	ErrVolumeInUse = errors.New("volume in use and cannot be removed")
)

// Volume represents a volume.
//
// See https://goo.gl/FZA4BK for more details.
type Volume struct {
	Name       string            `json:"Name" yaml:"Name"`
	Driver     string            `json:"Driver,omitempty" yaml:"Driver,omitempty"`
	Mountpoint string            `json:"Mountpoint,omitempty" yaml:"Mountpoint,omitempty"`
	Labels     map[string]string `json:"Labels,omitempty" yaml:"Labels,omitempty"`
}

// ListVolumesOptions specify parameters to the ListVolumes function.
//
// See https://goo.gl/FZA4BK for more details.
type ListVolumesOptions struct {
	Filters map[string][]string
	Context context.Context
}

// ListVolumes returns a list of available volumes in the server.
//
// See https://goo.gl/FZA4BK for more details.
func (c *Client) ListVolumes(opts ListVolumesOptions) ([]Volume, error) {
	resp, err := c.do("GET", "/volumes?"+queryString(opts), doOptions{
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	m := make(map[string]interface{})
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}
	var volumes []Volume
	volumesJSON, ok := m["Volumes"]
	if !ok {
		return volumes, nil
	}
	data, err := json.Marshal(volumesJSON)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &volumes); err != nil {
		return nil, err
	}
	return volumes, nil
}

// CreateVolumeOptions specify parameters to the CreateVolume function.
//
// See https://goo.gl/pBUbZ9 for more details.
type CreateVolumeOptions struct {
	Name       string
	Driver     string
	DriverOpts map[string]string
	Context    context.Context `json:"-"`
}

// CreateVolume creates a volume on the server.
//
// See https://goo.gl/pBUbZ9 for more details.
func (c *Client) CreateVolume(opts CreateVolumeOptions) (*Volume, error) {
	resp, err := c.do("POST", "/volumes/create", doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var volume Volume
	if err := json.NewDecoder(resp.Body).Decode(&volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// InspectVolume returns a volume by its name.
//
// See https://goo.gl/0g9A6i for more details.
func (c *Client) InspectVolume(name string) (*Volume, error) {
	resp, err := c.do("GET", "/volumes/"+name, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchVolume
		}
		return nil, err
	}
	defer resp.Body.Close()
	var volume Volume
	if err := json.NewDecoder(resp.Body).Decode(&volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// RemoveVolume removes a volume by its name.
//
// See https://goo.gl/79GNQz for more details.
func (c *Client) RemoveVolume(name string) error {
	resp, err := c.do("DELETE", "/volumes/"+name, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchVolume
			}
			if e.Status == http.StatusConflict {
				return ErrVolumeInUse
			}
		}
		return nil
	}
	defer resp.Body.Close()
	return nil
}
