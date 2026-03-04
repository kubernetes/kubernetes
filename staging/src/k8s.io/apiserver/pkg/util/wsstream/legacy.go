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

// Deprecated: This WebSockets package under apiserver is no longer in use.
// Please use the apimachinery version of the package at:
//
//	k8s.io/apimachinery/pkg/util/httpstream/wsstream
package wsstream

import apimachinerywsstream "k8s.io/apimachinery/pkg/util/httpstream/wsstream"

// Aliases for all exported symbols previously in "conn.go"
const (
	ChannelWebSocketProtocol       = apimachinerywsstream.ChannelWebSocketProtocol
	Base64ChannelWebSocketProtocol = apimachinerywsstream.Base64ChannelWebSocketProtocol
)

type ChannelType = apimachinerywsstream.ChannelType

const (
	IgnoreChannel    = apimachinerywsstream.IgnoreChannel
	ReadChannel      = apimachinerywsstream.ReadChannel
	WriteChannel     = apimachinerywsstream.WriteChannel
	ReadWriteChannel = apimachinerywsstream.ReadWriteChannel
)

type ChannelProtocolConfig = apimachinerywsstream.ChannelProtocolConfig

var (
	IsWebSocketRequest         = apimachinerywsstream.IsWebSocketRequest
	IgnoreReceives             = apimachinerywsstream.IgnoreReceives
	NewDefaultChannelProtocols = apimachinerywsstream.NewDefaultChannelProtocols
)

type Conn = apimachinerywsstream.Conn

var NewConn = apimachinerywsstream.NewConn

// Aliases for all exported symbols previously in "stream.go"
type ReaderProtocolConfig = apimachinerywsstream.ReaderProtocolConfig

var NewDefaultReaderProtocols = apimachinerywsstream.NewDefaultReaderProtocols

type Reader = apimachinerywsstream.Reader

var NewReader = apimachinerywsstream.NewReader
