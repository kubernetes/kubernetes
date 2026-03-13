// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

type grpcType string

// grpcType describes all types of grpc connection.
const (
	Unary        grpcType = "unary"
	ClientStream grpcType = "client_stream"
	ServerStream grpcType = "server_stream"
	BidiStream   grpcType = "bidi_stream"
)

// Kind describes whether interceptor is a client or server type.
type Kind string

// Enum for Client and Server Kind.
const (
	KindClient Kind = "client"
	KindServer Kind = "server"
)
