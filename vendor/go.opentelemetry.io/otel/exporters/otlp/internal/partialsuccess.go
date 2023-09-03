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

import "fmt"

// PartialSuccessDropKind indicates the kind of partial success error
// received by an OTLP exporter, which corresponds with the signal
// being exported.
type PartialSuccessDropKind string

const (
	// TracingPartialSuccess indicates that some spans were rejected.
	TracingPartialSuccess PartialSuccessDropKind = "spans"

	// MetricsPartialSuccess indicates that some metric data points were rejected.
	MetricsPartialSuccess PartialSuccessDropKind = "metric data points"
)

// PartialSuccess represents the underlying error for all handling
// OTLP partial success messages.  Use `errors.Is(err,
// PartialSuccess{})` to test whether an error passed to the OTel
// error handler belongs to this category.
type PartialSuccess struct {
	ErrorMessage  string
	RejectedItems int64
	RejectedKind  PartialSuccessDropKind
}

var _ error = PartialSuccess{}

// Error implements the error interface.
func (ps PartialSuccess) Error() string {
	msg := ps.ErrorMessage
	if msg == "" {
		msg = "empty message"
	}
	return fmt.Sprintf("OTLP partial success: %s (%d %s rejected)", msg, ps.RejectedItems, ps.RejectedKind)
}

// Is supports the errors.Is() interface.
func (ps PartialSuccess) Is(err error) bool {
	_, ok := err.(PartialSuccess)
	return ok
}

// PartialSuccessToError produces an error suitable for passing to
// `otel.Handle()` out of the fields in a partial success response,
// independent of which signal produced the outcome.
func PartialSuccessToError(kind PartialSuccessDropKind, itemsRejected int64, errorMessage string) error {
	return PartialSuccess{
		ErrorMessage:  errorMessage,
		RejectedItems: itemsRejected,
		RejectedKind:  kind,
	}
}
