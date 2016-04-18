/*
Copyright 2016 The Kubernetes Authors.

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
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wsstream"
)

const (
	stdinChannel = iota
	stdoutChannel
	stderrChannel
	errorChannel
	resizeChannel
)

// createChannels returns the standard channel types for a shell connection (STDIN 0, STDOUT 1, STDERR 2)
// along with the approximate duplex value. It also creates the error (3) and resize (4) channels.
func createChannels(opts *options) []wsstream.ChannelType {
	// open the requested channels, and always open the error channel
	channels := make([]wsstream.ChannelType, 5)
	channels[stdinChannel] = readChannel(opts.stdin)
	channels[stdoutChannel] = writeChannel(opts.stdout)
	channels[stderrChannel] = writeChannel(opts.stderr)
	channels[errorChannel] = wsstream.WriteChannel
	channels[resizeChannel] = wsstream.ReadChannel
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

// createWebSocketStreams returns a context containing the websocket connection and
// streams needed to perform an exec or an attach.
func createWebSocketStreams(req *http.Request, w http.ResponseWriter, opts *options, idleTimeout time.Duration) (*context, bool) {
	channels := createChannels(opts)
	conn := wsstream.NewConn(channels...)
	conn.SetIdleTimeout(idleTimeout)
	streams, err := conn.Open(httplog.Unlogged(w), req)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Unable to upgrade websocket connection: %v", err))
		return nil, false
	}

	// Send an empty message to the lowest writable channel to notify the client the connection is established
	// TODO: make generic to SPDY and WebSockets and do it outside of this method?
	switch {
	case opts.stdout:
		streams[stdoutChannel].Write([]byte{})
	case opts.stderr:
		streams[stderrChannel].Write([]byte{})
	default:
		streams[errorChannel].Write([]byte{})
	}

	return &context{
		conn:         conn,
		stdinStream:  streams[stdinChannel],
		stdoutStream: streams[stdoutChannel],
		stderrStream: streams[stderrChannel],
		errorStream:  streams[errorChannel],
		tty:          opts.tty,
		resizeStream: streams[resizeChannel],
	}, true
}
