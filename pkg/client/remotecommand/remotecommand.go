/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remotecommand

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion/queryparams"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream/spdy"
	"github.com/golang/glog"
)

type upgrader interface {
	upgrade(*client.Request, *client.Config) (httpstream.Connection, error)
}

type defaultUpgrader struct{}

func (u *defaultUpgrader) upgrade(req *client.Request, config *client.Config) (httpstream.Connection, error) {
	return req.Upgrade(config, spdy.NewRoundTripper)
}

type Streamer struct {
	req    *client.Request
	config *client.Config
	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
	tty    bool

	upgrader upgrader
}

// Executor executes a command on a pod container
type Executor struct {
	Streamer
	command []string
}

// New creates a new RemoteCommandExecutor
func New(req *client.Request, config *client.Config, command []string, stdin io.Reader, stdout, stderr io.Writer, tty bool) *Executor {
	return &Executor{
		command: command,
		Streamer: Streamer{
			req:    req,
			config: config,
			stdin:  stdin,
			stdout: stdout,
			stderr: stderr,
			tty:    tty,
		},
	}
}

type Attach struct {
	Streamer
}

// NewAttach creates a new RemoteAttach
func NewAttach(req *client.Request, config *client.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool) *Attach {
	return &Attach{
		Streamer: Streamer{
			req:    req,
			config: config,
			stdin:  stdin,
			stdout: stdout,
			stderr: stderr,
			tty:    tty,
		},
	}
}

// Execute sends a remote command execution request, upgrading the
// connection and creating streams to represent stdin/stdout/stderr. Data is
// copied between these streams and the supplied stdin/stdout/stderr parameters.
func (e *Attach) Execute() error {
	opts := api.PodAttachOptions{
		Stdin:  (e.stdin != nil),
		Stdout: (e.stdout != nil),
		Stderr: (!e.tty && e.stderr != nil),
		TTY:    e.tty,
	}

	if err := e.setupRequestParameters(&opts); err != nil {
		return err
	}

	return e.doStream()
}

// Execute sends a remote command execution request, upgrading the
// connection and creating streams to represent stdin/stdout/stderr. Data is
// copied between these streams and the supplied stdin/stdout/stderr parameters.
func (e *Executor) Execute() error {
	opts := api.PodExecOptions{
		Stdin:   (e.stdin != nil),
		Stdout:  (e.stdout != nil),
		Stderr:  (!e.tty && e.stderr != nil),
		TTY:     e.tty,
		Command: e.command,
	}

	if err := e.setupRequestParameters(&opts); err != nil {
		return err
	}

	return e.doStream()
}

func (e *Streamer) setupRequestParameters(obj runtime.Object) error {
	versioned, err := api.Scheme.ConvertToVersion(obj, e.config.Version)
	if err != nil {
		return err
	}
	params, err := queryparams.Convert(versioned)
	if err != nil {
		return err
	}
	for k, v := range params {
		for _, vv := range v {
			e.req.Param(k, vv)
		}
	}
	return nil
}

func (e *Streamer) doStream() error {
	if e.upgrader == nil {
		e.upgrader = &defaultUpgrader{}
	}
	conn, err := e.upgrader.upgrade(e.req, e.config)
	if err != nil {
		return err
	}
	defer conn.Close()

	doneChan := make(chan struct{}, 2)
	errorChan := make(chan error)

	cp := func(s string, dst io.Writer, src io.Reader) {
		glog.V(4).Infof("Copying %s", s)
		defer glog.V(4).Infof("Done copying %s", s)
		if _, err := io.Copy(dst, src); err != nil && err != io.EOF {
			glog.Errorf("Error copying %s: %v", s, err)
		}
		if s == api.StreamTypeStdout || s == api.StreamTypeStderr {
			doneChan <- struct{}{}
		}
	}

	headers := http.Header{}
	headers.Set(api.StreamType, api.StreamTypeError)
	errorStream, err := conn.CreateStream(headers)
	if err != nil {
		return err
	}
	go func() {
		message, err := ioutil.ReadAll(errorStream)
		if err != nil && err != io.EOF {
			errorChan <- fmt.Errorf("Error reading from error stream: %s", err)
			return
		}
		if len(message) > 0 {
			errorChan <- fmt.Errorf("Error executing remote command: %s", message)
			return
		}
	}()
	defer errorStream.Reset()

	if e.stdin != nil {
		headers.Set(api.StreamType, api.StreamTypeStdin)
		remoteStdin, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}
		defer remoteStdin.Reset()
		// TODO this goroutine will never exit cleanly (the io.Copy never unblocks)
		// because stdin is not closed until the process exits. If we try to call
		// stdin.Close(), it returns no error but doesn't unblock the copy. It will
		// exit when the process exits, instead.
		go cp(api.StreamTypeStdin, remoteStdin, e.stdin)
	}

	waitCount := 0
	completedStreams := 0

	if e.stdout != nil {
		waitCount++
		headers.Set(api.StreamType, api.StreamTypeStdout)
		remoteStdout, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}
		defer remoteStdout.Reset()
		go cp(api.StreamTypeStdout, e.stdout, remoteStdout)
	}

	if e.stderr != nil && !e.tty {
		waitCount++
		headers.Set(api.StreamType, api.StreamTypeStderr)
		remoteStderr, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}
		defer remoteStderr.Reset()
		go cp(api.StreamTypeStderr, e.stderr, remoteStderr)
	}

Loop:
	for {
		select {
		case <-doneChan:
			completedStreams++
			if completedStreams == waitCount {
				break Loop
			}
		case err := <-errorChan:
			return err
		}
	}

	return nil
}
