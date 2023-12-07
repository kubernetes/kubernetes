// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/otlp/otlptrace/otlpconfig/optiontypes.go.tmpl

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

package otlpconfig // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/otlpconfig"

const (
	// DefaultCollectorGRPCPort is the default gRPC port of the collector.
	DefaultCollectorGRPCPort uint16 = 4317
	// DefaultCollectorHTTPPort is the default HTTP port of the collector.
	DefaultCollectorHTTPPort uint16 = 4318
	// DefaultCollectorHost is the host address the Exporter will attempt
	// connect to if no collector address is provided.
	DefaultCollectorHost string = "localhost"
)

// Compression describes the compression used for payloads sent to the
// collector.
type Compression int

const (
	// NoCompression tells the driver to send payloads without
	// compression.
	NoCompression Compression = iota
	// GzipCompression tells the driver to send payloads after
	// compressing them with gzip.
	GzipCompression
)

// Marshaler describes the kind of message format sent to the collector.
type Marshaler int

const (
	// MarshalProto tells the driver to send using the protobuf binary format.
	MarshalProto Marshaler = iota
	// MarshalJSON tells the driver to send using json format.
	MarshalJSON
)
