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
	"net/url"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
)

// Executor is an interface for transporting shell-style streams.
type Executor interface {
	// Stream initiates the transport of the standard shell streams. It will transport any
	// non-nil stream to a remote system, and return an error if a problem occurs. If tty
	// is set, the stderr stream is not used (raw TTY manages stdout and stderr over the
	// stdout stream).
	Stream(stdin io.Reader, stdout, stderr io.Writer, tty bool) error
}

// StreamExecutor supports the ability to dial an httpstream connection and the ability to
// run a command line stream protocol over that dialer.
type StreamExecutor interface {
	Executor
	httpstream.Dialer
}

// streamExecutor handles transporting standard shell streams over an httpstream connection.
type streamExecutor struct {
	upgrader  httpstream.UpgradeRoundTripper
	transport http.RoundTripper

	method string
	url    *url.URL
}

// NewExecutor connects to the provided server and upgrades the connection to
// multiplexed bidirectional streams. The current implementation uses SPDY,
// but this could be replaced with HTTP/2 once it's available, or something else.
// TODO: the common code between this and portforward could be abstracted.
func NewExecutor(config *client.Config, method string, url *url.URL) (StreamExecutor, error) {
	tlsConfig, err := client.TLSConfigFor(config)
	if err != nil {
		return nil, err
	}

	upgradeRoundTripper := spdy.NewRoundTripper(tlsConfig)
	wrapper, err := client.HTTPWrappersForConfig(config, upgradeRoundTripper)
	if err != nil {
		return nil, err
	}

	return &streamExecutor{
		upgrader:  upgradeRoundTripper,
		transport: wrapper,
		method:    method,
		url:       url,
	}, nil
}

// NewStreamExecutor upgrades the request so that it supports multiplexed bidirectional
// streams. This method takes a stream upgrader and an optional function that is invoked
// to wrap the round tripper. This method may be used by clients that are lower level than
// Kubernetes clients or need to provide their own upgrade round tripper.
func NewStreamExecutor(upgrader httpstream.UpgradeRoundTripper, fn func(http.RoundTripper) http.RoundTripper, method string, url *url.URL) (StreamExecutor, error) {
	var rt http.RoundTripper = upgrader
	if fn != nil {
		rt = fn(rt)
	}
	return &streamExecutor{
		upgrader:  upgrader,
		transport: rt,
		method:    method,
		url:       url,
	}, nil
}

// Dial opens a connection to a remote server and attempts to negotiate a SPDY connection.
func (e *streamExecutor) Dial() (httpstream.Connection, error) {
	client := &http.Client{Transport: e.transport}

	req, err := http.NewRequest(e.method, e.url.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %s", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %s", err)
	}
	defer resp.Body.Close()

	// TODO: handle protocol selection in the future
	return e.upgrader.NewConnection(resp)
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	conn, err := e.Dial()
	if err != nil {
		return err
	}
	defer conn.Close()
	// TODO: negotiate protocols
	streamer := &streamProtocol{
		stdin:  stdin,
		stdout: stdout,
		stderr: stderr,
		tty:    tty,
	}
	return streamer.stream(conn)
}

type streamProtocol struct {
	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
	tty    bool
}

func (e *streamProtocol) stream(conn httpstream.Connection) error {
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
