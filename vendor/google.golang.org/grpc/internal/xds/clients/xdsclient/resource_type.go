/*
 *
 * Copyright 2025 gRPC authors.
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

package xdsclient

// ResourceType wraps all resource-type specific functionality. Each supported
// resource type needs to provide an implementation of the Decoder.
type ResourceType struct {
	// TypeURL is the xDS type URL of this resource type for the v3 xDS
	// protocol. This URL is used as the key to look up the corresponding
	// ResourceType implementation in the ResourceTypes map provided in the
	// Config.
	TypeURL string

	// TODO: Revisit if we need TypeURL to be part of the struct because it is
	// already a key in config's ResouceTypes map.

	// TypeName is a shorter representation of the TypeURL to identify the
	// resource type. It is used for logging/debugging purposes.
	TypeName string

	// AllResourcesRequiredInSotW indicates whether this resource type requires
	// that all resources be present in every SotW response from the server. If
	// true, a response that does not include a previously seen resource will
	// be interpreted as a deletion of that resource.
	AllResourcesRequiredInSotW bool

	// Decoder is used to deserialize and validate an xDS resource received
	// from the xDS management server.
	Decoder Decoder
}

// Decoder wraps the resource-type specific functionality for validation
// and deserialization.
type Decoder interface {
	// Decode deserializes and validates an xDS resource as received from the
	// xDS management server.
	//
	// The `resource` parameter may contain a value of the serialized wrapped
	// resource (i.e. with the type URL
	// `type.googleapis.com/envoy.service.discovery.v3.Resource`).
	// Implementations are responsible for unwrapping the underlying resource if
	// it is wrapped.
	//
	// If unmarshalling or validation fails, it returns a non-nil error.
	// Otherwise, returns a fully populated DecodeResult.
	Decode(resource AnyProto, options DecodeOptions) (*DecodeResult, error)
}

// AnyProto contains the type URL and serialized proto data of an xDS resource.
type AnyProto struct {
	TypeURL string
	Value   []byte
}

// DecodeOptions wraps the options required by ResourceType implementations for
// decoding configuration received from the xDS management server.
type DecodeOptions struct {
	// Config contains the complete configuration passed to the xDS client.
	// This contains useful data for resource validation.
	Config *Config

	// ServerConfig contains the configuration of the xDS server that provided
	// the current resource being decoded.
	ServerConfig *ServerConfig
}

// DecodeResult is the result of a decode operation.
type DecodeResult struct {
	// Name is the name of the decoded resource.
	Name string

	// Resource contains the configuration associated with the decoded
	// resource.
	Resource ResourceData
}

// ResourceData contains the configuration data sent by the xDS management
// server, associated with the resource being watched. Every resource type must
// provide an implementation of this interface to represent the configuration
// received from the xDS management server.
type ResourceData interface {
	// Equal returns true if the passed in resource data is equal to that of
	// the receiver.
	Equal(other ResourceData) bool

	// Bytes returns the underlying raw bytes of the resource proto.
	Bytes() []byte
}
