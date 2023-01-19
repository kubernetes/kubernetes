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

// Package internal contains common functionality for all OTLP exporters.
package internal // import "go.opentelemetry.io/otel/exporters/otlp/internal"

import "go.opentelemetry.io/otel"

// GetUserAgentHeader return an OTLP header value form "OTel OTLP Exporter Go/{{ .Version }}"
// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#user-agent
func GetUserAgentHeader() string {
	return "OTel OTLP Exporter Go/" + otel.Version()
}
