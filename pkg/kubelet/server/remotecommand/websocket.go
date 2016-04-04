/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/util/wsstream"

	"github.com/golang/glog"
)

// standardShellChannels returns the standard channel types for a shell connection (STDIN 0, STDOUT 1, STDERR 2)
// along with the approximate duplex value. Supported subprotocols are "channel.k8s.io" and
// "base64.channel.k8s.io".
func standardShellChannels(stdin, stdout, stderr bool) []wsstream.ChannelType {
	// open three half-duplex channels
	channels := []wsstream.ChannelType{wsstream.ReadChannel, wsstream.WriteChannel, wsstream.WriteChannel}
	if !stdin {
		channels[0] = wsstream.IgnoreChannel
	}
	if !stdout {
		channels[1] = wsstream.IgnoreChannel
	}
	if !stderr {
		channels[2] = wsstream.IgnoreChannel
	}
	return channels
}

// createWebSocketStreams returns a remoteCommandContext containing the websocket connection and
// streams needed to perform an exec or an attach.
func createWebSocketStreams(req *http.Request, w http.ResponseWriter, opts *options, idleTimeout time.Duration) (*context, bool) {
	// open the requested channels, and always open the error channel
	channels := append(standardShellChannels(opts.stdin, opts.stdout, opts.stderr), wsstream.WriteChannel)
	conn := wsstream.NewConn(channels...)
	conn.SetIdleTimeout(idleTimeout)
	streams, err := conn.Open(httplog.Unlogged(w), req)
	if err != nil {
		glog.Errorf("Unable to upgrade websocket connection: %v", err)
		return nil, false
	}
	// Send an empty message to the lowest writable channel to notify the client the connection is established
	// TODO: make generic to SPDY and WebSockets and do it outside of this method?
	switch {
	case opts.stdout:
		streams[1].Write([]byte{})
	case opts.stderr:
		streams[2].Write([]byte{})
	default:
		streams[3].Write([]byte{})
	}
	return &context{
		conn:         conn,
		stdinStream:  streams[0],
		stdoutStream: streams[1],
		stderrStream: streams[2],
		errorStream:  streams[3],
		tty:          opts.tty,
	}, true
}
