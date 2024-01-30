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

// Package wsstream contains utilities for streaming content over WebSockets.
// The Conn type allows callers to multiplex multiple read/write channels over
// a single websocket.
//
// "channel.k8s.io"
//
// The Websocket RemoteCommand subprotocol "channel.k8s.io" prepends each binary message with a
// byte indicating the channel number (zero indexed) the message was sent on. Messages in both
// directions should prefix their messages with this channel byte. Used for remote execution,
// the channel numbers are by convention defined to match the POSIX file-descriptors assigned
// to STDIN, STDOUT, and STDERR (0, 1, and 2). No other conversion is performed on the raw
// subprotocol - writes are sent as they are received by the server.
//
// Example client session:
//
//	CONNECT http://server.com with subprotocol "channel.k8s.io"
//	WRITE []byte{0, 102, 111, 111, 10} # send "foo\n" on channel 0 (STDIN)
//	READ  []byte{1, 10}                # receive "\n" on channel 1 (STDOUT)
//	CLOSE
//
// "v2.channel.k8s.io"
//
// The second Websocket subprotocol version "v2.channel.k8s.io" is the same as version 1,
// but it is the first "versioned" subprotocol.
//
// "v3.channel.k8s.io"
//
// The third version of the Websocket RemoteCommand subprotocol adds another channel
// for terminal resizing events. This channel is prepended with the byte '3', and it
// transmits two window sizes (encoding TerminalSize struct) with integers in the range
// (0,65536].
//
// "v4.channel.k8s.io"
//
// The fourth version of the Websocket RemoteCommand subprotocol adds a channel for
// errors. This channel returns structured errors containing process exit codes. The
// error is "apierrors.StatusError{}".
//
// "v5.channel.k8s.io"
//
// The fifth version of the Websocket RemoteCommand subprotocol adds a CLOSE signal,
// which is sent as the first byte of the message. The second byte is the channel
// id. This CLOSE signal is handled by the websocket server by closing the stream,
// allowing the other streams to complete transmission if necessary, and gracefully
// shutdown the connection.
//
// Example client session:
//
//	CONNECT http://server.com with subprotocol "v5.channel.k8s.io"
//	WRITE []byte{0, 102, 111, 111, 10} # send "foo\n" on channel 0 (STDIN)
//	WRITE []byte{255, 0}               # send CLOSE signal (STDIN)
//	CLOSE
package wsstream // import "k8s.io/apimachinery/pkg/util/httpstream/wsstream"
