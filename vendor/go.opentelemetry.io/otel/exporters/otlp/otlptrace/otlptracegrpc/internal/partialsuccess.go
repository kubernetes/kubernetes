// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/otlp/partialsuccess.go

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

package internal // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal"

import "fmt"

// PartialSuccess represents the underlying error for all handling
// OTLP partial success messages.  Use `errors.Is(err,
// PartialSuccess{})` to test whether an error passed to the OTel
// error handler belongs to this category.
type PartialSuccess struct {
	ErrorMessage  string
	RejectedItems int64
	RejectedKind  string
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

// TracePartialSuccessError returns an error describing a partial success
// response for the trace signal.
func TracePartialSuccessError(itemsRejected int64, errorMessage string) error {
	return PartialSuccess{
		ErrorMessage:  errorMessage,
		RejectedItems: itemsRejected,
		RejectedKind:  "spans",
	}
}

// MetricPartialSuccessError returns an error describing a partial success
// response for the metric signal.
func MetricPartialSuccessError(itemsRejected int64, errorMessage string) error {
	return PartialSuccess{
		ErrorMessage:  errorMessage,
		RejectedItems: itemsRejected,
		RejectedKind:  "metric data points",
	}
}
