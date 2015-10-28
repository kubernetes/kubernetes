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
	"net/http"
	"net/url"

	"github.com/golang/glog"

	client "k8s.io/kubernetes/pkg/client/unversioned"
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

// Dial opens a connection to a remote server and attempts to negotiate a SPDY
// connection. Upon success, it returns the connection and the protocol
// selected by the server.
func (e *streamExecutor) Dial(protocols ...string) (httpstream.Connection, string, error) {
	transport := e.transport
	// TODO consider removing this and reusing client.TransportFor above to get this for free
	switch {
	case bool(glog.V(9)):
		transport = client.NewDebuggingRoundTripper(transport, client.CurlCommand, client.URLTiming, client.ResponseHeaders)
	case bool(glog.V(8)):
		transport = client.NewDebuggingRoundTripper(transport, client.JustURL, client.RequestHeaders, client.ResponseStatus, client.ResponseHeaders)
	case bool(glog.V(7)):
		transport = client.NewDebuggingRoundTripper(transport, client.JustURL, client.RequestHeaders, client.ResponseStatus)
	case bool(glog.V(6)):
		transport = client.NewDebuggingRoundTripper(transport, client.URLTiming)
	}

	// TODO the client probably shouldn't be created here, as it doesn't allow
	// flexibility to allow callers to configure it.
	client := &http.Client{Transport: transport}

	req, err := http.NewRequest(e.method, e.url.String(), nil)
	if err != nil {
		return nil, "", fmt.Errorf("error creating request: %v", err)
	}
	for i := range protocols {
		req.Header.Add(httpstream.HeaderProtocolVersion, protocols[i])
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("error sending request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusSwitchingProtocols {
		return nil, "", fmt.Errorf("unexpected response status code %d (%s)", resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	conn, err := e.upgrader.NewConnection(resp)
	if err != nil {
		return nil, "", err
	}

	return conn, resp.Header.Get(httpstream.HeaderProtocolVersion), nil
}

const (
	// The SPDY subprotocol "channel.k8s.io" is used for remote command
	// attachment/execution. This represents the initial unversioned subprotocol,
	// which has the known bugs http://issues.k8s.io/13394 and
	// http://issues.k8s.io/13395.
	StreamProtocolV1Name = "channel.k8s.io"
	// The SPDY subprotocol "v2.channel.k8s.io" is used for remote command
	// attachment/execution. It is the second version of the subprotocol and
	// resolves the issues present in the first version.
	StreamProtocolV2Name = "v2.channel.k8s.io"
)

type streamProtocolHandler interface {
	stream(httpstream.Connection) error
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	supportedProtocols := []string{StreamProtocolV2Name, StreamProtocolV1Name}
	conn, protocol, err := e.Dial(supportedProtocols...)
	if err != nil {
		return err
	}
	defer conn.Close()

	var streamer streamProtocolHandler

	switch protocol {
	case StreamProtocolV2Name:
		streamer = &streamProtocolV2{
			stdin:  stdin,
			stdout: stdout,
			stderr: stderr,
			tty:    tty,
		}
	case "":
		glog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", StreamProtocolV1Name)
		fallthrough
	case StreamProtocolV1Name:
		streamer = &streamProtocolV1{
			stdin:  stdin,
			stdout: stdout,
			stderr: stderr,
			tty:    tty,
		}
	}

	return streamer.stream(conn)
}
