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

package internal // import "go.opentelemetry.io/otel/exporters/otlp/internal"

// ErrorKind is used to identify the kind of export error
// being wrapped.
type ErrorKind int

const (
	// TracesExport indicates the error comes from the OTLP trace exporter.
	TracesExport ErrorKind = iota
)

// prefix returns a prefix for the Error() string.
func (k ErrorKind) prefix() string {
	switch k {
	case TracesExport:
		return "traces export: "
	default:
		return "unknown: "
	}
}

// wrappedExportError wraps an OTLP exporter error with the kind of
// signal that produced it.
type wrappedExportError struct {
	wrap error
	kind ErrorKind
}

// WrapTracesError wraps an error from the OTLP exporter for traces.
func WrapTracesError(err error) error {
	return wrappedExportError{
		wrap: err,
		kind: TracesExport,
	}
}

var _ error = wrappedExportError{}

// Error attaches a prefix corresponding to the kind of exporter.
func (t wrappedExportError) Error() string {
	return t.kind.prefix() + t.wrap.Error()
}

// Unwrap returns the wrapped error.
func (t wrappedExportError) Unwrap() error {
	return t.wrap
}
