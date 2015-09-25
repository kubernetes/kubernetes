// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"errors"
	"net/http"
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
	Name       string `json:"Name" yaml:"Name"`
	Driver     string `json:"Driver,omitempty" yaml:"Driver,omitempty"`
	Mountpoint string `json:"Mountpoint,omitempty" yaml:"Mountpoint,omitempty"`
}

// ListVolumesOptions specify parameters to the ListVolumes function.
//
// See https://goo.gl/FZA4BK for more details.
type ListVolumesOptions struct {
	Filters map[string][]string
}

// ListVolumes returns a list of available volumes in the server.
//
// See https://goo.gl/FZA4BK for more details.
func (c *Client) ListVolumes(opts ListVolumesOptions) ([]Volume, error) {
	body, _, err := c.do("GET", "/volumes?"+queryString(opts), doOptions{})
	if err != nil {
		return nil, err
	}
	m := make(map[string]interface{})
	if err := json.Unmarshal(body, &m); err != nil {
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
}

// CreateVolume creates a volume on the server.
//
// See https://goo.gl/pBUbZ9 for more details.
func (c *Client) CreateVolume(opts CreateVolumeOptions) (*Volume, error) {
	body, _, err := c.do("POST", "/volumes", doOptions{data: opts})
	if err != nil {
		return nil, err
	}
	var volume Volume
	if err := json.Unmarshal(body, &volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// InspectVolume returns a volume by its name.
//
// See https://goo.gl/0g9A6i for more details.
func (c *Client) InspectVolume(name string) (*Volume, error) {
	body, status, err := c.do("GET", "/volumes/"+name, doOptions{})
	if status == http.StatusNotFound {
		return nil, ErrNoSuchVolume
	}
	if err != nil {
		return nil, err
	}
	var volume Volume
	if err := json.Unmarshal(body, &volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// RemoveVolume removes a volume by its name.
//
// See https://goo.gl/79GNQz for more details.
func (c *Client) RemoveVolume(name string) error {
	_, status, err := c.do("DELETE", "/volumes/"+name, doOptions{})
	if status == http.StatusNotFound {
		return ErrNoSuchVolume
	}
	if status == http.StatusConflict {
		return ErrVolumeInUse
	}
	return err
}
