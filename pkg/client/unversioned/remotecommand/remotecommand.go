/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
)

// StreamOptions holds information pertaining to the current streaming session: supported stream
// protocols, input/output streams, if the client is requesting a TTY, and a terminal size queue to
// support terminal resizing.
type StreamOptions struct {
	SupportedProtocols []string
	Stdin              io.Reader
	Stdout             io.Writer
	Stderr             io.Writer
	Tty                bool
	TerminalSizeQueue  TerminalSizeQueue
}

// Executor is an interface for transporting shell-style streams.
type Executor interface {
	// Stream initiates the transport of the standard shell streams. It will transport any
	// non-nil stream to a remote system, and return an error if a problem occurs. If tty
	// is set, the stderr stream is not used (raw TTY manages stdout and stderr over the
	// stdout stream).
	Stream(options StreamOptions) error
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
func NewExecutor(config *restclient.Config, method string, url *url.URL) (StreamExecutor, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, err
	}

	upgradeRoundTripper := spdy.NewRoundTripper(tlsConfig, true)
	wrapper, err := restclient.HTTPWrappersForConfig(config, upgradeRoundTripper)
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
	rt := http.RoundTripper(upgrader)
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
	rt := transport.DebugWrappers(e.transport)

	// TODO the client probably shouldn't be created here, as it doesn't allow
	// flexibility to allow callers to configure it.
	client := &http.Client{Transport: rt}

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

	conn, err := e.upgrader.NewConnection(resp)
	if err != nil {
		return nil, "", err
	}

	return conn, resp.Header.Get(httpstream.HeaderProtocolVersion), nil
}

type streamCreator interface {
	CreateStream(headers http.Header) (httpstream.Stream, error)
}

type streamProtocolHandler interface {
	stream(conn streamCreator) error
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(options StreamOptions) error {
	conn, protocol, err := e.Dial(options.SupportedProtocols...)
	if err != nil {
		return err
	}
	defer conn.Close()

	var streamer streamProtocolHandler

	switch protocol {
	case remotecommand.StreamProtocolV4Name:
		streamer = newStreamProtocolV4(options)
	case remotecommand.StreamProtocolV3Name:
		streamer = newStreamProtocolV3(options)
	case remotecommand.StreamProtocolV2Name:
		streamer = newStreamProtocolV2(options)
	case "":
		glog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", remotecommand.StreamProtocolV1Name)
		fallthrough
	case remotecommand.StreamProtocolV1Name:
		streamer = newStreamProtocolV1(options)
	}

	return streamer.stream(conn)
}
