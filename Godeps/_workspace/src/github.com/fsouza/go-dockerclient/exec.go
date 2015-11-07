// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

// Exec is the type representing a `docker exec` instance and containing the
// instance ID
type Exec struct {
	ID string `json:"Id,omitempty" yaml:"Id,omitempty"`
}

// CreateExecOptions specify parameters to the CreateExecContainer function.
//
// See https://goo.gl/1KSIb7 for more details
type CreateExecOptions struct {
	AttachStdin  bool     `json:"AttachStdin,omitempty" yaml:"AttachStdin,omitempty"`
	AttachStdout bool     `json:"AttachStdout,omitempty" yaml:"AttachStdout,omitempty"`
	AttachStderr bool     `json:"AttachStderr,omitempty" yaml:"AttachStderr,omitempty"`
	Tty          bool     `json:"Tty,omitempty" yaml:"Tty,omitempty"`
	Cmd          []string `json:"Cmd,omitempty" yaml:"Cmd,omitempty"`
	Container    string   `json:"Container,omitempty" yaml:"Container,omitempty"`
	User         string   `json:"User,omitempty" yaml:"User,omitempty"`
}

// CreateExec sets up an exec instance in a running container `id`, returning the exec
// instance, or an error in case of failure.
//
// See https://goo.gl/1KSIb7 for more details
func (c *Client) CreateExec(opts CreateExecOptions) (*Exec, error) {
	path := fmt.Sprintf("/containers/%s/exec", opts.Container)
	resp, err := c.do("POST", path, doOptions{data: opts})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, &NoSuchContainer{ID: opts.Container}
		}
		return nil, err
	}
	defer resp.Body.Close()
	var exec Exec
	if err := json.NewDecoder(resp.Body).Decode(&exec); err != nil {
		return nil, err
	}

	return &exec, nil
}

// StartExecOptions specify parameters to the StartExecContainer function.
//
// See https://goo.gl/iQCnto for more details
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

// StartExec starts a previously set up exec instance id. If opts.Detach is
// true, it returns after starting the exec command. Otherwise, it sets up an
// interactive session with the exec command.
//
// See https://goo.gl/iQCnto for more details
func (c *Client) StartExec(id string, opts StartExecOptions) error {
	if id == "" {
		return &NoSuchExec{ID: id}
	}

	path := fmt.Sprintf("/exec/%s/start", id)

	if opts.Detach {
		resp, err := c.do("POST", path, doOptions{data: opts})
		if err != nil {
			if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
				return &NoSuchExec{ID: id}
			}
			return err
		}
		defer resp.Body.Close()
		return nil
	}

	return c.hijack("POST", path, hijackOptions{
		success:        opts.Success,
		setRawTerminal: opts.RawTerminal,
		in:             opts.InputStream,
		stdout:         opts.OutputStream,
		stderr:         opts.ErrorStream,
		data:           opts,
	})
}

// ResizeExecTTY resizes the tty session used by the exec command id. This API
// is valid only if Tty was specified as part of creating and starting the exec
// command.
//
// See https://goo.gl/e1JpsA for more details
func (c *Client) ResizeExecTTY(id string, height, width int) error {
	params := make(url.Values)
	params.Set("h", strconv.Itoa(height))
	params.Set("w", strconv.Itoa(width))

	path := fmt.Sprintf("/exec/%s/resize?%s", id, params.Encode())
	resp, err := c.do("POST", path, doOptions{})
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// ExecProcessConfig is a type describing the command associated to a Exec
// instance. It's used in the ExecInspect type.
type ExecProcessConfig struct {
	Privileged bool     `json:"privileged,omitempty" yaml:"privileged,omitempty"`
	User       string   `json:"user,omitempty" yaml:"user,omitempty"`
	Tty        bool     `json:"tty,omitempty" yaml:"tty,omitempty"`
	EntryPoint string   `json:"entrypoint,omitempty" yaml:"entrypoint,omitempty"`
	Arguments  []string `json:"arguments,omitempty" yaml:"arguments,omitempty"`
}

// ExecInspect is a type with details about a exec instance, including the
// exit code if the command has finished running. It's returned by a api
// call to /exec/(id)/json
//
// See https://goo.gl/gPtX9R for more details
type ExecInspect struct {
	ID            string            `json:"ID,omitempty" yaml:"ID,omitempty"`
	Running       bool              `json:"Running,omitempty" yaml:"Running,omitempty"`
	ExitCode      int               `json:"ExitCode,omitempty" yaml:"ExitCode,omitempty"`
	OpenStdin     bool              `json:"OpenStdin,omitempty" yaml:"OpenStdin,omitempty"`
	OpenStderr    bool              `json:"OpenStderr,omitempty" yaml:"OpenStderr,omitempty"`
	OpenStdout    bool              `json:"OpenStdout,omitempty" yaml:"OpenStdout,omitempty"`
	ProcessConfig ExecProcessConfig `json:"ProcessConfig,omitempty" yaml:"ProcessConfig,omitempty"`
	Container     Container         `json:"Container,omitempty" yaml:"Container,omitempty"`
}

// InspectExec returns low-level information about the exec command id.
//
// See https://goo.gl/gPtX9R for more details
func (c *Client) InspectExec(id string) (*ExecInspect, error) {
	path := fmt.Sprintf("/exec/%s/json", id)
	resp, err := c.do("GET", path, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, &NoSuchExec{ID: id}
		}
		return nil, err
	}
	defer resp.Body.Close()
	var exec ExecInspect
	if err := json.NewDecoder(resp.Body).Decode(&exec); err != nil {
		return nil, err
	}
	return &exec, nil
}

// NoSuchExec is the error returned when a given exec instance does not exist.
type NoSuchExec struct {
	ID string
}

func (err *NoSuchExec) Error() string {
	return "No such exec instance: " + err.ID
}
