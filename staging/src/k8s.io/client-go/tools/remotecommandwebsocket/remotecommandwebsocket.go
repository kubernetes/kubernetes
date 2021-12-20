/*
Copyright 2020 The Kubernetes Authors.

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

package remotecommandwebsocket

import (
	"io"
	"net/http"
	"net/url"
	"time"

	gwebsocket "github.com/gorilla/websocket"
	"k8s.io/apimachinery/pkg/util/httpstream"
	ws "k8s.io/apimachinery/pkg/util/httpstream/websocket"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	remotecommandspdy "k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

const (
	// StreamStdIn represents the remote stdin stream
	StreamStdIn = 0
	// StreamStdOut represents the remote stdout stream
	StreamStdOut = 1
	// StreamStdErr represents the remote stderr stream
	StreamStdErr = 2
	// StreamErr respresents the remote error stream
	StreamErr = 3
	// StreamResize represents the resize stream
	StreamResize = 4

	// Base64StreamStdIn represents the remote stdin stream
	Base64StreamStdIn = 48
	// Base64StreamStdOut represents the remote stdout stream
	Base64StreamStdOut = 49
	// Base64StreamStdErr represents the remote stderr stream
	Base64StreamStdErr = 50
	// Base64StreamErr respresents the remote error stream
	Base64StreamErr = 51
	// Base64StreamResize represents the resize stream
	Base64StreamResize = 52

	// WebSocketExitStream is the error code that represents a clean exit from the stream
	WebSocketExitStream = 1000

	// Time allowed to write a message to the peer.
	writeWait = 10 * time.Second

	// Maximum message size allowed from peer.
	maxMessageSize = 8192

	// Time allowed to read the next pong message from the peer.
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait.
	pingPeriod = (pongWait * 9) / 10

	// Time to wait before force close on connection.
	closeGracePeriod = 10 * time.Second

	preV4BinaryWebsocketProtocol = "channel.k8s.io"
	preV4Base64WebsocketProtocol = "base64.channel.k8s.io"
	v4BinaryWebsocketProtocol    = "v4." + preV4BinaryWebsocketProtocol
	v4Base64WebsocketProtocol    = "v4." + preV4Base64WebsocketProtocol
)

// StreamOptions holds information pertaining to the current streaming session:
// input/output streams, if the client is requesting a TTY, and a terminal size queue to
// support terminal resizing.
type StreamOptions struct {
	Stdin             io.Reader
	Stdout            io.Writer
	Stderr            io.Writer
	Tty               bool
	TerminalSizeQueue remotecommandspdy.TerminalSizeQueue
}

// Executor is an interface for transporting shell-style streams.
type Executor interface {
	// Stream initiates the transport of the standard shell streams. It will transport any
	// non-nil stream to a remote system, and return an error if a problem occurs. If tty
	// is set, the stderr stream is not used (raw TTY manages stdout and stderr over the
	// stdout stream).
	Stream(options StreamOptions) error
}

// streamExecutor handles transporting standard shell streams over an httpstream connection.
type streamExecutor struct {
	upgrader  websocket.Upgrader
	transport http.RoundTripper

	wsRoundTripper ws.RoundTripper

	url       *url.URL
	protocols []string
}

type streamProtocolHandler interface {
	stream(conn *gwebsocket.Conn) error
}

// NewWebSocketExecutor creates a new websocket connection to the URL specified with
// the rest client's TLS configuration and headers
func NewWebSocketExecutor(config *restclient.Config, url *url.URL) (Executor, error) {

	if url.Scheme == "https" {
		url.Scheme = "wss"
	} else if url.Scheme == "http" {
		url.Scheme = "ws"
	}

	wrapper, upgradeRoundTripper, err := websocket.RoundTripperFor(config)
	if err != nil {
		return nil, err
	}

	return NewWebSocketExecutorForTransports(wrapper, upgradeRoundTripper, url)
}

// NewWebSocketExecutorForTransports connects to the provided server using the given transport,
// upgrades the response using the given upgrader to multiplexed bidirectional streams.
func NewWebSocketExecutorForTransports(transport http.RoundTripper, upgrader websocket.Upgrader, url *url.URL) (Executor, error) {
	return NewWebSocketExecutorForProtocols(
		transport, upgrader, url,
		v4BinaryWebsocketProtocol,
	)

	//remotecommand.StreamProtocolV4Name,
	//	remotecommand.StreamProtocolV3Name,
	//	remotecommand.StreamProtocolV2Name,
}

// NewWebSocketExecutorForProtocols connects to the provided server and upgrades the connection to
// multiplexed bidirectional streams using only the provided protocols. Exposed for testing, most
// callers should use NewWebSocketExecutor or NewWebSocketExecutorForTransports.
func NewWebSocketExecutorForProtocols(transport http.RoundTripper, upgrader websocket.Upgrader, url *url.URL, protocols ...string) (Executor, error) {
	return &streamExecutor{
		upgrader:  upgrader,
		transport: transport,
		url:       url,
		protocols: protocols,
	}, nil
}

// Stream opens a protocol streamer to the server and streams until a client closes
// the connection or the server disconnects.
func (e *streamExecutor) Stream(options StreamOptions) error {
	// Leverage the existing rest tools to get a connection with the corrrecet
	// TLS and headers
	req, err := http.NewRequest(httpstream.HeaderUpgrade, e.url.String(), nil)

	con, protocol, err := websocket.Negotiate(
		e.upgrader,
		&http.Client{Transport: e.transport},
		req,
		e.protocols...,
	)
	if err != nil {
		return err
	}

	// cast the connection to a websocket to get the underlying connection
	conn, ok := con.(*ws.Connection)

	if !ok {
		panic("Connection is not a websocket connection")
	}

	var streamer streamProtocolHandler

	klog.V(4).Infof("The protocol is  %s", protocol)

	switch protocol {
	case v4BinaryWebsocketProtocol:
		streamer = newBinaryV4(options)
	case v4Base64WebsocketProtocol:
		streamer = newBase64V4(options)
	case preV4Base64WebsocketProtocol:
		streamer = newPreV4Base64Protocol(options)
	case "":
		klog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", remotecommand.StreamProtocolV1Name)
		fallthrough
	case preV4BinaryWebsocketProtocol:
		streamer = newPreV4BinaryProtocol(options)
	}

	return streamer.stream(conn.Conn)

}
