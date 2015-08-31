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
	"sync"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/conversion/queryparams"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
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

	headers := http.Header{}

	// set up error stream
	errorChan := make(chan error)
	headers.Set(api.StreamType, api.StreamTypeError)
	errorStream, err := conn.CreateStream(headers)
	if err != nil {
		return err
	}

	go func() {
		message, err := ioutil.ReadAll(errorStream)
		switch {
		case err != nil && err != io.EOF:
			errorChan <- fmt.Errorf("error reading from error stream: %s", err)
		case len(message) > 0:
			errorChan <- fmt.Errorf("error executing remote command: %s", message)
		default:
			errorChan <- nil
		}
		close(errorChan)
	}()

	var wg sync.WaitGroup
	var once sync.Once

	// set up stdin stream
	if e.stdin != nil {
		headers.Set(api.StreamType, api.StreamTypeStdin)
		remoteStdin, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}

		// copy from client's stdin to container's stdin
		go func() {
			// if e.stdin is noninteractive, e.g. `echo abc | kubectl exec -i <pod> -- cat`, make sure
			// we close remoteStdin as soon as the copy from e.stdin to remoteStdin finishes. Otherwise
			// the executed command will remain running.
			defer once.Do(func() { remoteStdin.Close() })

			if _, err := io.Copy(remoteStdin, e.stdin); err != nil {
				util.HandleError(err)
			}
		}()

		// read from remoteStdin until the stream is closed. this is essential to
		// be able to exit interactive sessions cleanly and not leak goroutines or
		// hang the client's terminal.
		//
		// go-dockerclient's current hijack implementation
		// (https://github.com/fsouza/go-dockerclient/blob/89f3d56d93788dfe85f864a44f85d9738fca0670/client.go#L564)
		// waits for all three streams (stdin/stdout/stderr) to finish copying
		// before returning. When hijack finishes copying stdout/stderr, it calls
		// Close() on its side of remoteStdin, which allows this copy to complete.
		// When that happens, we must Close() on our side of remoteStdin, to
		// allow the copy in hijack to complete, and hijack to return.
		go func() {
			defer once.Do(func() { remoteStdin.Close() })
			// this "copy" doesn't actually read anything - it's just here to wait for
			// the server to close remoteStdin.
			if _, err := io.Copy(ioutil.Discard, remoteStdin); err != nil {
				util.HandleError(err)
			}
		}()
	}

	// set up stdout stream
	if e.stdout != nil {
		headers.Set(api.StreamType, api.StreamTypeStdout)
		remoteStdout, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := io.Copy(e.stdout, remoteStdout); err != nil {
				util.HandleError(err)
			}
		}()
	}

	// set up stderr stream
	if e.stderr != nil && !e.tty {
		headers.Set(api.StreamType, api.StreamTypeStderr)
		remoteStderr, err := conn.CreateStream(headers)
		if err != nil {
			return err
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := io.Copy(e.stderr, remoteStderr); err != nil {
				util.HandleError(err)
			}
		}()
	}

	// we're waiting for stdout/stderr to finish copying
	wg.Wait()

	// waits for errorStream to finish reading with an error or nil
	return <-errorChan
}
