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

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import "context"

// SpanExporter handles the delivery of spans to external receivers. This is
// the final component in the trace export pipeline.
type SpanExporter interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// ExportSpans exports a batch of spans.
	//
	// This function is called synchronously, so there is no concurrency
	// safety requirement. However, due to the synchronous calling pattern,
	// it is critical that all timeouts and cancellations contained in the
	// passed context must be honored.
	//
	// Any retry logic must be contained in this function. The SDK that
	// calls this function will not implement any retry logic. All errors
	// returned by this function are considered unrecoverable and will be
	// reported to a configured error Handler.
	ExportSpans(ctx context.Context, spans []ReadOnlySpan) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Shutdown notifies the exporter of a pending halt to operations. The
	// exporter is expected to preform any cleanup or synchronization it
	// requires while honoring all timeouts and cancellations contained in
	// the passed context.
	Shutdown(ctx context.Context) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}
