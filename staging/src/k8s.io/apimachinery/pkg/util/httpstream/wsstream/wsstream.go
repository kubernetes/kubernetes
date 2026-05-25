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

package wsstream

import (
	"io"
	"net/http"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/klog/v2"
	streamws "k8s.io/streaming/pkg/httpstream/wsstream"
)

const (
	WebSocketProtocolHeader        = streamws.WebSocketProtocolHeader
	ChannelWebSocketProtocol       = streamws.ChannelWebSocketProtocol
	Base64ChannelWebSocketProtocol = streamws.Base64ChannelWebSocketProtocol
)

type ChannelType = streamws.ChannelType

const (
	IgnoreChannel    = streamws.IgnoreChannel
	ReadChannel      = streamws.ReadChannel
	WriteChannel     = streamws.WriteChannel
	ReadWriteChannel = streamws.ReadWriteChannel
)

func IsWebSocketRequest(req *http.Request) bool {
	return streamws.IsWebSocketRequest(req)
}

func IsWebSocketRequestWithStreamCloseProtocol(req *http.Request) bool {
	return streamws.IsWebSocketRequestWithStreamCloseProtocol(req)
}

func IsWebSocketRequestWithTunnelingProtocol(req *http.Request) bool {
	return streamws.IsWebSocketRequestWithTunnelingProtocol(req)
}

func IgnoreReceives(ws *websocket.Conn, timeout time.Duration) {
	streamws.IgnoreReceives(ws, timeout)
}

func IgnoreReceivesWithLogger(logger klog.Logger, ws *websocket.Conn, timeout time.Duration) {
	streamws.IgnoreReceivesWithLogger(logger, ws, timeout)
}

type ChannelProtocolConfig = streamws.ChannelProtocolConfig

func NewDefaultChannelProtocols(channels []ChannelType) map[string]ChannelProtocolConfig {
	return streamws.NewDefaultChannelProtocols(channels)
}

type Conn = streamws.Conn

func NewConn(protocols map[string]ChannelProtocolConfig) *Conn {
	return streamws.NewConn(protocols)
}

type ReaderProtocolConfig = streamws.ReaderProtocolConfig

func NewDefaultReaderProtocols() map[string]ReaderProtocolConfig {
	return streamws.NewDefaultReaderProtocols()
}

type Reader = streamws.Reader

func NewReader(r io.Reader, ping bool, protocols map[string]ReaderProtocolConfig) *Reader {
	return streamws.NewReader(r, ping, protocols)
}

func NewReaderWithLogger(logger klog.Logger, r io.Reader, ping bool, protocols map[string]ReaderProtocolConfig) *Reader {
	return streamws.NewReaderWithLogger(logger, r, ping, protocols)
}
