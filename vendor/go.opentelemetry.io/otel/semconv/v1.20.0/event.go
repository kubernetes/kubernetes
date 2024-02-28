// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.20.0"

import "go.opentelemetry.io/otel/attribute"

// This semantic convention defines the attributes used to represent a feature
// flag evaluation as an event.
const (
	// FeatureFlagKeyKey is the attribute Key conforming to the
	// "feature_flag.key" semantic conventions. It represents the unique
	// identifier of the feature flag.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'logo-color'
	FeatureFlagKeyKey = attribute.Key("feature_flag.key")

	// FeatureFlagProviderNameKey is the attribute Key conforming to the
	// "feature_flag.provider_name" semantic conventions. It represents the
	// name of the service provider that performs the flag evaluation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'Flag Manager'
	FeatureFlagProviderNameKey = attribute.Key("feature_flag.provider_name")

	// FeatureFlagVariantKey is the attribute Key conforming to the
	// "feature_flag.variant" semantic conventions. It represents the sHOULD be
	// a semantic identifier for a value. If one is unavailable, a stringified
	// version of the value can be used.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'red', 'true', 'on'
	// Note: A semantic identifier, commonly referred to as a variant, provides
	// a means
	// for referring to a value without including the value itself. This can
	// provide additional context for understanding the meaning behind a value.
	// For example, the variant `red` maybe be used for the value `#c05543`.
	//
	// A stringified version of the value can be used in situations where a
	// semantic identifier is unavailable. String representation of the value
	// should be determined by the implementer.
	FeatureFlagVariantKey = attribute.Key("feature_flag.variant")
)

// FeatureFlagKey returns an attribute KeyValue conforming to the
// "feature_flag.key" semantic conventions. It represents the unique identifier
// of the feature flag.
func FeatureFlagKey(val string) attribute.KeyValue {
	return FeatureFlagKeyKey.String(val)
}

// FeatureFlagProviderName returns an attribute KeyValue conforming to the
// "feature_flag.provider_name" semantic conventions. It represents the name of
// the service provider that performs the flag evaluation.
func FeatureFlagProviderName(val string) attribute.KeyValue {
	return FeatureFlagProviderNameKey.String(val)
}

// FeatureFlagVariant returns an attribute KeyValue conforming to the
// "feature_flag.variant" semantic conventions. It represents the sHOULD be a
// semantic identifier for a value. If one is unavailable, a stringified
// version of the value can be used.
func FeatureFlagVariant(val string) attribute.KeyValue {
	return FeatureFlagVariantKey.String(val)
}

// RPC received/sent message.
const (
	// MessageTypeKey is the attribute Key conforming to the "message.type"
	// semantic conventions. It represents the whether this is a received or
	// sent message.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	MessageTypeKey = attribute.Key("message.type")

	// MessageIDKey is the attribute Key conforming to the "message.id"
	// semantic conventions. It represents the mUST be calculated as two
	// different counters starting from `1` one for sent messages and one for
	// received message.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Note: This way we guarantee that the values will be consistent between
	// different implementations.
	MessageIDKey = attribute.Key("message.id")

	// MessageCompressedSizeKey is the attribute Key conforming to the
	// "message.compressed_size" semantic conventions. It represents the
	// compressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	MessageCompressedSizeKey = attribute.Key("message.compressed_size")

	// MessageUncompressedSizeKey is the attribute Key conforming to the
	// "message.uncompressed_size" semantic conventions. It represents the
	// uncompressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	MessageUncompressedSizeKey = attribute.Key("message.uncompressed_size")
)

var (
	// sent
	MessageTypeSent = MessageTypeKey.String("SENT")
	// received
	MessageTypeReceived = MessageTypeKey.String("RECEIVED")
)

// MessageID returns an attribute KeyValue conforming to the "message.id"
// semantic conventions. It represents the mUST be calculated as two different
// counters starting from `1` one for sent messages and one for received
// message.
func MessageID(val int) attribute.KeyValue {
	return MessageIDKey.Int(val)
}

// MessageCompressedSize returns an attribute KeyValue conforming to the
// "message.compressed_size" semantic conventions. It represents the compressed
// size of the message in bytes.
func MessageCompressedSize(val int) attribute.KeyValue {
	return MessageCompressedSizeKey.Int(val)
}

// MessageUncompressedSize returns an attribute KeyValue conforming to the
// "message.uncompressed_size" semantic conventions. It represents the
// uncompressed size of the message in bytes.
func MessageUncompressedSize(val int) attribute.KeyValue {
	return MessageUncompressedSizeKey.Int(val)
}

// The attributes used to report a single exception associated with a span.
const (
	// ExceptionEscapedKey is the attribute Key conforming to the
	// "exception.escaped" semantic conventions. It represents the sHOULD be
	// set to true if the exception event is recorded at a point where it is
	// known that the exception is escaping the scope of the span.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	// Note: An exception is considered to have escaped (or left) the scope of
	// a span,
	// if that span is ended while the exception is still logically "in
	// flight".
	// This may be actually "in flight" in some languages (e.g. if the
	// exception
	// is passed to a Context manager's `__exit__` method in Python) but will
	// usually be caught at the point of recording the exception in most
	// languages.
	//
	// It is usually not possible to determine at the point where an exception
	// is thrown
	// whether it will escape the scope of a span.
	// However, it is trivial to know that an exception
	// will escape, if one checks for an active exception just before ending
	// the span,
	// as done in the [example above](#recording-an-exception).
	//
	// It follows that an exception may still escape the scope of the span
	// even if the `exception.escaped` attribute was not set or set to false,
	// since the event might have been recorded at a time where it was not
	// clear whether the exception will escape.
	ExceptionEscapedKey = attribute.Key("exception.escaped")
)

// ExceptionEscaped returns an attribute KeyValue conforming to the
// "exception.escaped" semantic conventions. It represents the sHOULD be set to
// true if the exception event is recorded at a point where it is known that
// the exception is escaping the scope of the span.
func ExceptionEscaped(val bool) attribute.KeyValue {
	return ExceptionEscapedKey.Bool(val)
}
