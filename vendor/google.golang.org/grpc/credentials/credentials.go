/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package credentials implements various credentials supported by gRPC library,
// which encapsulate all the state needed by a client to authenticate with a
// server and make various assertions, e.g., about the client's identity, role,
// or whether it is authorized to make a particular call.
package credentials // import "google.golang.org/grpc/credentials"

import (
	"context"
	"errors"
	"net"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc/internal"
)

// PerRPCCredentials defines the common interface for the credentials which need to
// attach security information to every RPC (e.g., oauth2).
type PerRPCCredentials interface {
	// GetRequestMetadata gets the current request metadata, refreshing
	// tokens if required. This should be called by the transport layer on
	// each request, and the data should be populated in headers or other
	// context. If a status code is returned, it will be used as the status
	// for the RPC. uri is the URI of the entry point for the request.
	// When supported by the underlying implementation, ctx can be used for
	// timeout and cancellation. Additionally, RequestInfo data will be
	// available via ctx to this call.
	// TODO(zhaoq): Define the set of the qualified keys instead of leaving
	// it as an arbitrary string.
	GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error)
	// RequireTransportSecurity indicates whether the credentials requires
	// transport security.
	RequireTransportSecurity() bool
}

// ProtocolInfo provides information regarding the gRPC wire protocol version,
// security protocol, security protocol version in use, server name, etc.
type ProtocolInfo struct {
	// ProtocolVersion is the gRPC wire protocol version.
	ProtocolVersion string
	// SecurityProtocol is the security protocol in use.
	SecurityProtocol string
	// SecurityVersion is the security protocol version.
	SecurityVersion string
	// ServerName is the user-configured server name.
	ServerName string
}

// AuthInfo defines the common interface for the auth information the users are interested in.
type AuthInfo interface {
	AuthType() string
}

// ErrConnDispatched indicates that rawConn has been dispatched out of gRPC
// and the caller should not close rawConn.
var ErrConnDispatched = errors.New("credentials: rawConn is dispatched out of gRPC")

// TransportCredentials defines the common interface for all the live gRPC wire
// protocols and supported transport security protocols (e.g., TLS, SSL).
type TransportCredentials interface {
	// ClientHandshake does the authentication handshake specified by the corresponding
	// authentication protocol on rawConn for clients. It returns the authenticated
	// connection and the corresponding auth information about the connection.
	// Implementations must use the provided context to implement timely cancellation.
	// gRPC will try to reconnect if the error returned is a temporary error
	// (io.EOF, context.DeadlineExceeded or err.Temporary() == true).
	// If the returned error is a wrapper error, implementations should make sure that
	// the error implements Temporary() to have the correct retry behaviors.
	//
	// If the returned net.Conn is closed, it MUST close the net.Conn provided.
	ClientHandshake(context.Context, string, net.Conn) (net.Conn, AuthInfo, error)
	// ServerHandshake does the authentication handshake for servers. It returns
	// the authenticated connection and the corresponding auth information about
	// the connection.
	//
	// If the returned net.Conn is closed, it MUST close the net.Conn provided.
	ServerHandshake(net.Conn) (net.Conn, AuthInfo, error)
	// Info provides the ProtocolInfo of this TransportCredentials.
	Info() ProtocolInfo
	// Clone makes a copy of this TransportCredentials.
	Clone() TransportCredentials
	// OverrideServerName overrides the server name used to verify the hostname on the returned certificates from the server.
	// gRPC internals also use it to override the virtual hosting name if it is set.
	// It must be called before dialing. Currently, this is only used by grpclb.
	OverrideServerName(string) error
}

// Bundle is a combination of TransportCredentials and PerRPCCredentials.
//
// It also contains a mode switching method, so it can be used as a combination
// of different credential policies.
//
// Bundle cannot be used together with individual TransportCredentials.
// PerRPCCredentials from Bundle will be appended to other PerRPCCredentials.
//
// This API is experimental.
type Bundle interface {
	TransportCredentials() TransportCredentials
	PerRPCCredentials() PerRPCCredentials
	// NewWithMode should make a copy of Bundle, and switch mode. Modifying the
	// existing Bundle may cause races.
	//
	// NewWithMode returns nil if the requested mode is not supported.
	NewWithMode(mode string) (Bundle, error)
}

// RequestInfo contains request data attached to the context passed to GetRequestMetadata calls.
//
// This API is experimental.
type RequestInfo struct {
	// The method passed to Invoke or NewStream for this RPC. (For proto methods, this has the format "/some.Service/Method")
	Method string
}

// requestInfoKey is a struct to be used as the key when attaching a RequestInfo to a context object.
type requestInfoKey struct{}

// RequestInfoFromContext extracts the RequestInfo from the context if it exists.
//
// This API is experimental.
func RequestInfoFromContext(ctx context.Context) (ri RequestInfo, ok bool) {
	ri, ok = ctx.Value(requestInfoKey{}).(RequestInfo)
	return
}

func init() {
	internal.NewRequestInfoContext = func(ctx context.Context, ri RequestInfo) context.Context {
		return context.WithValue(ctx, requestInfoKey{}, ri)
	}
}

// ChannelzSecurityInfo defines the interface that security protocols should implement
// in order to provide security info to channelz.
//
// This API is experimental.
type ChannelzSecurityInfo interface {
	GetSecurityValue() ChannelzSecurityValue
}

// ChannelzSecurityValue defines the interface that GetSecurityValue() return value
// should satisfy. This interface should only be satisfied by *TLSChannelzSecurityValue
// and *OtherChannelzSecurityValue.
//
// This API is experimental.
type ChannelzSecurityValue interface {
	isChannelzSecurityValue()
}

// OtherChannelzSecurityValue defines the struct that non-TLS protocol should return
// from GetSecurityValue(), which contains protocol specific security info. Note
// the Value field will be sent to users of channelz requesting channel info, and
// thus sensitive info should better be avoided.
//
// This API is experimental.
type OtherChannelzSecurityValue struct {
	ChannelzSecurityValue
	Name  string
	Value proto.Message
}
