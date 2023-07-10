/*
Copyright 2023 The Kubernetes Authors.

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

package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	constants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/klog/v2"
)

const defaultIdleConnectionTimeout = 4 * time.Hour

// Options contains details about which streams are required for
// remote command execution.
type Options struct {
	Stdin  bool
	Stdout bool
	Stderr bool
	Tty    bool
}

// conns contains the connection and streams used when
// forwarding an attach or execute session into a container.
type conns struct {
	conn         io.Closer
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	writeStatus  func(status *apierrors.StatusError) error
	resizeStream io.ReadCloser
	resizeChan   chan remotecommand.TerminalSize
	tty          bool
}

// Create WebSocket server streams to respond to a WebSocket client. Creates the streams passed
// in the stream options.
func webSocketServerStreams(req *http.Request, w http.ResponseWriter, opts Options) (*conns, error) {
	ctx, err := createWebSocketStreams(req, w, opts)
	if err != nil {
		return nil, err
	}

	if ctx.resizeStream != nil {
		ctx.resizeChan = make(chan remotecommand.TerminalSize)
		go handleResizeEvents(req.Context(), ctx.resizeStream, ctx.resizeChan)
	}

	return ctx, nil
}

// Read terminal resize events off of passed stream and queue into passed channel.
func handleResizeEvents(ctx context.Context, stream io.Reader, channel chan<- remotecommand.TerminalSize) {
	defer runtime.HandleCrash()
	defer close(channel)

	decoder := json.NewDecoder(stream)
	for {
		size := remotecommand.TerminalSize{}
		if err := decoder.Decode(&size); err != nil {
			break
		}

		select {
		case channel <- size:
		case <-ctx.Done():
			// To avoid leaking this routine, exit if the http request finishes. This path
			// would generally be hit if starting the process fails and nothing is started to
			// ingest these resize events.
			return
		}
	}
}

// createChannels returns the standard channel types for a shell connection (STDIN 0, STDOUT 1, STDERR 2)
// along with the approximate duplex value. It also creates the error (3) and resize (4) channels.
func createChannels(opts Options) []wsstream.ChannelType {
	// open the requested channels, and always open the error channel
	channels := make([]wsstream.ChannelType, 5)
	channels[constants.StreamStdIn] = readChannel(opts.Stdin)
	channels[constants.StreamStdOut] = writeChannel(opts.Stdout)
	channels[constants.StreamStdErr] = writeChannel(opts.Stderr)
	channels[constants.StreamErr] = wsstream.WriteChannel
	channels[constants.StreamResize] = wsstream.ReadChannel
	return channels
}

// readChannel returns wsstream.ReadChannel if real is true, or wsstream.IgnoreChannel.
func readChannel(real bool) wsstream.ChannelType {
	if real {
		return wsstream.ReadChannel
	}
	return wsstream.IgnoreChannel
}

// writeChannel returns wsstream.WriteChannel if real is true, or wsstream.IgnoreChannel.
func writeChannel(real bool) wsstream.ChannelType {
	if real {
		return wsstream.WriteChannel
	}
	return wsstream.IgnoreChannel
}

// createWebSocketStreams returns a "conns" struct containing the websocket connection and
// streams needed to perform an exec or an attach.
func createWebSocketStreams(req *http.Request, w http.ResponseWriter, opts Options) (*conns, error) {
	channels := createChannels(opts)
	conn := wsstream.NewConn(map[string]wsstream.ChannelProtocolConfig{
		// WebSocket server only supports remote command version 5.
		constants.StreamProtocolV5Name: {
			Binary:   true,
			Channels: channels,
		},
	})
	conn.SetIdleTimeout(defaultIdleConnectionTimeout)
	// Opening the connection responds to WebSocket client, negotiating
	// the WebSocket upgrade connection and the subprotocol.
	negotiatedProtocol, streams, err := conn.Open(w, req)
	if err != nil {
		runtime.HandleError(fmt.Errorf("unable to upgrade websocket connection: %v", err))
		return nil, err
	}
	klog.V(4).Infof("websocket server negotiated sub-protocol: %s", negotiatedProtocol)

	// Send an empty message to the lowest writable channel to notify the client the connection is established
	switch {
	case opts.Stdout:
		streams[constants.StreamStdOut].Write([]byte{})
	case opts.Stderr:
		streams[constants.StreamStdErr].Write([]byte{})
	default:
		streams[constants.StreamErr].Write([]byte{})
	}

	ctx := &conns{
		conn:         conn,
		stdinStream:  streams[constants.StreamStdIn],
		stdoutStream: streams[constants.StreamStdOut],
		stderrStream: streams[constants.StreamStdErr],
		tty:          opts.Tty,
		resizeStream: streams[constants.StreamResize],
	}

	ctx.writeStatus = v4WriteStatusFunc(streams[constants.StreamErr])

	return ctx, nil
}

// v4WriteStatusFunc returns a WriteStatusFunc that marshals a given api Status
// as json in the error channel.
func v4WriteStatusFunc(stream io.Writer) func(status *apierrors.StatusError) error {
	return func(status *apierrors.StatusError) error {
		bs, err := json.Marshal(status.Status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}
