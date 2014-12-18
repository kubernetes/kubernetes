// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Docs can currently be found at https://github.com/docker/docker/blob/master/docs/sources/reference/api/docker_remote_api_v1.15.md#exec-create

package docker

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

// CreateExecOptions specify parameters to the CreateExecContainer function.
//
// See http://goo.gl/8izrzI for more details
type CreateExecOptions struct {
	AttachStdin  bool     `json:"AttachStdin,omitempty" yaml:"AttachStdin,omitempty"`
	AttachStdout bool     `json:"AttachStdout,omitempty" yaml:"AttachStdout,omitempty"`
	AttachStderr bool     `json:"AttachStderr,omitempty" yaml:"AttachStderr,omitempty"`
	Tty          bool     `json:"Tty,omitempty" yaml:"Tty,omitempty"`
	Cmd          []string `json:"Cmd,omitempty" yaml:"Cmd,omitempty"`
	Container    string   `json:"Container,omitempty" yaml:"Container,omitempty"`
}

// StartExecOptions specify parameters to the StartExecContainer function.
//
// See http://goo.gl/JW8Lxl for more details
type StartExecOptions struct {
	Detach bool `json:"Detach,omitempty" yaml:"Detach,omitempty"`

	Tty bool `json:"Tty,omitempty" yaml:"Tty,omitempty"`

	InputStream  io.Reader `qs:"-"`
	OutputStream io.Writer `qs:"-"`
	ErrorStream  io.Writer `qs:"-"`

	// Use raw terminal? Usually true when the container contains a TTY.
	RawTerminal bool `qs:"-"`

	// If set, after a successful connect, a sentinel will be sent and then the
	// client will block on receive before continuing.
	//
	// It must be an unbuffered channel. Using a buffered channel can lead
	// to unexpected behavior.
	Success chan struct{} `json:"-"`
}

// Exec is the type representing a `docker exec` instance and containing the
// instance ID
type Exec struct {
	ID string `json:"Id,omitempty" yaml:"Id,omitempty"`
}

// CreateExec sets up an exec instance in a running container `id`, returning the exec
// instance, or an error in case of failure.
//
// See http://goo.gl/8izrzI for more details
func (c *Client) CreateExec(opts CreateExecOptions) (*Exec, error) {
	path := fmt.Sprintf("/containers/%s/exec", opts.Container)
	body, status, err := c.do("POST", path, opts)
	if status == http.StatusNotFound {
		return nil, &NoSuchContainer{ID: opts.Container}
	}
	if err != nil {
		return nil, err
	}
	var exec Exec
	err = json.Unmarshal(body, &exec)
	if err != nil {
		return nil, err
	}

	return &exec, nil
}

// StartExec starts a previously set up exec instance id. If opts.Detach is
// true, it returns after starting the exec command. Otherwise, it sets up an
// interactive session with the exec command.
//
// See http://goo.gl/JW8Lxl for more details
func (c *Client) StartExec(id string, opts StartExecOptions) error {
	if id == "" {
		return &NoSuchExec{ID: id}
	}

	path := fmt.Sprintf("/exec/%s/start", id)

	if opts.Detach {
		_, status, err := c.do("POST", path, opts)
		if status == http.StatusNotFound {
			return &NoSuchExec{ID: id}
		}
		if err != nil {
			return err
		}
		return nil
	}

	return c.hijack("POST", path, opts.Success, opts.RawTerminal, opts.InputStream, opts.ErrorStream, opts.OutputStream, opts)
}

// ResizeExecTTY resizes the tty session used by the exec command id. This API
// is valid only if Tty was specified as part of creating and starting the exec
// command.
//
// See http://goo.gl/YDSx1f for more details
func (c *Client) ResizeExecTTY(id string, height, width int) error {
	params := make(url.Values)
	params.Set("h", strconv.Itoa(height))
	params.Set("w", strconv.Itoa(width))

	path := fmt.Sprintf("/exec/%s/resize?%s", id, params.Encode())
	_, _, err := c.do("POST", path, nil)
	return err
}

// NoSuchExec is the error returned when a given exec instance does not exist.
type NoSuchExec struct {
	ID string
}

func (err *NoSuchExec) Error() string {
	return "No such exec instance: " + err.ID
}
