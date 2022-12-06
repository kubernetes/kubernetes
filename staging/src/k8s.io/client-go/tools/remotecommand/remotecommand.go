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
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport/spdy"
)

// StreamOptions holds information pertaining to the current streaming session:
// input/output streams, if the client is requesting a TTY, and a terminal size queue to
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
	// Deprecated: use StreamWithContext instead to avoid possible resource leaks.
	// See https://github.com/kubernetes/kubernetes/pull/103177 for details.
	Stream(options StreamOptions) error

	// StreamWithContext initiates the transport of the standard shell streams. It will
	// transport any non-nil stream to a remote system, and return an error if a problem
	// occurs. If tty is set, the stderr stream is not used (raw TTY manages stdout and
	// stderr over the stdout stream).
	// The context controls the entire lifetime of stream execution.
	StreamWithContext(ctx context.Context, options StreamOptions) error
}

type streamCreator interface {
	CreateStream(headers http.Header) (httpstream.Stream, error)
}

type streamProtocolHandler interface {
	stream(conn streamCreator) error
}

// streamExecutor handles transporting standard shell streams over an httpstream connection.
type streamExecutor struct {
	upgrader  spdy.Upgrader
	transport http.RoundTripper

	method    string
	url       *url.URL
	protocols []string
}

// NewSPDYExecutor connects to the provided server and upgrades the connection to
// multiplexed bidirectional streams.
func NewSPDYExecutor(config *restclient.Config, method string, url *url.URL) (Executor, error) {
	wrapper, upgradeRoundTripper, err := spdy.RoundTripperFor(config)
	if err != nil {
		return nil, err
	}
	return NewSPDYExecutorForTransports(wrapper, upgradeRoundTripper, method, url)
}

// NewSPDYExecutorForTransports connects to the provided server using the given transport,
// upgrades the response using the given upgrader to multiplexed bidirectional streams.
func NewSPDYExecutorForTransports(transport http.RoundTripper, upgrader spdy.Upgrader, method string, url *url.URL) (Executor, error) {
	return NewSPDYExecutorForProtocols(
		transport, upgrader, method, url,
		remotecommand.StreamProtocolV4Name,
		remotecommand.StreamProtocolV3Name,
		remotecommand.StreamProtocolV2Name,
		remotecommand.StreamProtocolV1Name,
	)
}

// NewSPDYExecutorForProtocols connects to the provided server and upgrades the connection to
// multiplexed bidirectional streams using only the provided protocols. Exposed for testing, most
// callers should use NewSPDYExecutor or NewSPDYExecutorForTransports.
func NewSPDYExecutorForProtocols(transport http.RoundTripper, upgrader spdy.Upgrader, method string, url *url.URL, protocols ...string) (Executor, error) {
	return &streamExecutor{
		upgrader:  upgrader,
		transport: transport,
		method:    method,
		url:       url,
		protocols: protocols,
	}, nil
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(options StreamOptions) error {
	return e.StreamWithContext(context.Background(), options)
}

// newConnectionAndStream creates a new SPDY connection and a stream protocol handler upon it.
func (e *streamExecutor) newConnectionAndStream(ctx context.Context, options StreamOptions) (httpstream.Connection, streamProtocolHandler, error) {
	req, err := http.NewRequestWithContext(ctx, e.method, e.url.String(), nil)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating request: %v", err)
	}

	conn, protocol, err := spdy.Negotiate(
		e.upgrader,
		&http.Client{Transport: e.transport},
		req,
		e.protocols...,
	)
	if err != nil {
		return nil, nil, err
	}

	var streamer streamProtocolHandler

	switch protocol {
	case remotecommand.StreamProtocolV4Name:
		streamer = newStreamProtocolV4(options)
	case remotecommand.StreamProtocolV3Name:
		streamer = newStreamProtocolV3(options)
	case remotecommand.StreamProtocolV2Name:
		streamer = newStreamProtocolV2(options)
	case "":
		klog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", remotecommand.StreamProtocolV1Name)
		fallthrough
	case remotecommand.StreamProtocolV1Name:
		streamer = newStreamProtocolV1(options)
	}

	return conn, streamer, nil
}

// StreamWithContext opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects or the context is done.
func (e *streamExecutor) StreamWithContext(ctx context.Context, options StreamOptions) error {
	conn, streamer, err := e.newConnectionAndStream(ctx, options)
	if err != nil {
		return err
	}
	defer conn.Close()

	panicChan := make(chan any, 1)
	errorChan := make(chan error, 1)
	go func() {
		defer func() {
			if p := recover(); p != nil {
				panicChan <- p
			}
		}()
		errorChan <- streamer.stream(conn)
	}()

	select {
	case p := <-panicChan:
		panic(p)
	case err := <-errorChan:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
