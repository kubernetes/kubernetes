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
	Stdin             io.Reader
	Stdout            io.Writer
	Stderr            io.Writer
	Tty               bool
	TerminalSizeQueue TerminalSizeQueue
}

// Executor is an interface for transporting shell-style streams.
type Executor interface {
	// Stream initiates the transport of the standard shell streams. It will transport any
	// non-nil stream to a remote system, and return an error if a problem occurs. If tty
	// is set, the stderr stream is not used (raw TTY manages stdout and stderr over the
	// stdout stream).
	Stream(options StreamOptions) error
}

// SPDYUpgrader validates a response from the server after a SPDY upgrade.
type SPDYUpgrader interface {
	// NewConnection validates the response and creates a new Connection.
	NewConnection(resp *http.Response) (httpstream.Connection, error)
}

type streamCreator interface {
	CreateStream(headers http.Header) (httpstream.Stream, error)
}

type streamProtocolHandler interface {
	stream(conn streamCreator) error
}

// streamExecutor handles transporting standard shell streams over an httpstream connection.
type streamExecutor struct {
	upgrader  SPDYUpgrader
	transport http.RoundTripper

	method string
	url    *url.URL
}

// NewSPDYExecutor connects to the provided server and upgrades the connection to
// multiplexed bidirectional streams.
func NewSPDYExecutor(config *restclient.Config, method string, url *url.URL) (Executor, error) {
	wrapper, upgradeRoundTripper, err := SPDYRoundTripperFor(config)
	if err != nil {
		return nil, err
	}
	wrapper = transport.DebugWrappers(wrapper)
	return &streamExecutor{
		upgrader:  upgradeRoundTripper,
		transport: wrapper,
		method:    method,
		url:       url,
	}, nil
}

type spdyDialer struct {
	client   *http.Client
	upgrader SPDYUpgrader
	method   string
	url      *url.URL
}

func NewSPDYDialer(upgrader SPDYUpgrader, client *http.Client, method string, url *url.URL) (httpstream.Dialer, error) {
	return &spdyDialer{
		client:   client,
		upgrader: upgrader,
		method:   method,
		url:      url,
	}, nil
}

func (d *spdyDialer) Dial(protocols ...string) (httpstream.Connection, string, error) {
	req, err := http.NewRequest(d.method, d.url.String(), nil)
	if err != nil {
		return nil, "", fmt.Errorf("error creating request: %v", err)
	}
	return NegotiateSPDYConnection(d.upgrader, d.client, req, protocols...)
}

// SPDYRoundTripperFor returns a round tripper to use with SPDY.
func SPDYRoundTripperFor(config *restclient.Config) (http.RoundTripper, SPDYUpgrader, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, nil, err
	}
	upgradeRoundTripper := spdy.NewRoundTripper(tlsConfig, true)
	wrapper, err := restclient.HTTPWrappersForConfig(config, upgradeRoundTripper)
	if err != nil {
		return nil, nil, err
	}
	return wrapper, upgradeRoundTripper, nil
}

// NegotiateSPDYConnection opens a connection to a remote server and attempts to negotiate
// a SPDY connection. Upon success, it returns the connection and the protocol selected by
// the server. The client transport must use the upgradeRoundTripper - see SPDYRoundTripperFor.
func NegotiateSPDYConnection(upgrader SPDYUpgrader, client *http.Client, req *http.Request, protocols ...string) (httpstream.Connection, string, error) {
	for i := range protocols {
		req.Header.Add(httpstream.HeaderProtocolVersion, protocols[i])
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("error sending request: %v", err)
	}
	defer resp.Body.Close()
	conn, err := upgrader.NewConnection(resp)
	if err != nil {
		return nil, "", err
	}
	return conn, resp.Header.Get(httpstream.HeaderProtocolVersion), nil
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(options StreamOptions) error {
	req, err := http.NewRequest(e.method, e.url.String(), nil)
	if err != nil {
		return fmt.Errorf("error creating request: %v", err)
	}

	conn, protocol, err := NegotiateSPDYConnection(
		e.upgrader,
		&http.Client{Transport: e.transport},
		req,
		remotecommand.StreamProtocolV4Name,
		remotecommand.StreamProtocolV3Name,
		remotecommand.StreamProtocolV2Name,
		remotecommand.StreamProtocolV1Name,
	)
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
