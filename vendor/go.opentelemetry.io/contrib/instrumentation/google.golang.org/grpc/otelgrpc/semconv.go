// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

import (
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

// Semantic conventions for attribute keys for gRPC.
const (
	// Name of message transmitted or received.
	RPCNameKey = attribute.Key("name")

	// Type of message transmitted or received.
	RPCMessageTypeKey = attribute.Key("message.type")

	// Identifier of message transmitted or received.
	RPCMessageIDKey = attribute.Key("message.id")

	// The compressed size of the message transmitted or received in bytes.
	RPCMessageCompressedSizeKey = attribute.Key("message.compressed_size")

	// The uncompressed size of the message transmitted or received in
	// bytes.
	RPCMessageUncompressedSizeKey = attribute.Key("message.uncompressed_size")
)

// Semantic conventions for common RPC attributes.
var (
	// Semantic convention for gRPC as the remoting system.
	RPCSystemGRPC = semconv.RPCSystemGRPC

	// Semantic convention for a message named message.
	RPCNameMessage = RPCNameKey.String("message")

	// Semantic conventions for RPC message types.
	RPCMessageTypeSent     = RPCMessageTypeKey.String("SENT")
	RPCMessageTypeReceived = RPCMessageTypeKey.String("RECEIVED")
)
