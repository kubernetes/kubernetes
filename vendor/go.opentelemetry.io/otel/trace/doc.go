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

/*
Package trace provides an implementation of the tracing part of the
OpenTelemetry API.

To participate in distributed traces a Span needs to be created for the
operation being performed as part of a traced workflow. It its simplest form:

	var tracer trace.Tracer

	func init() {
		tracer = otel.Tracer("instrumentation/package/name")
	}

	func operation(ctx context.Context) {
		var span trace.Span
		ctx, span = tracer.Start(ctx, "operation")
		defer span.End()
		// ...
	}

A Tracer is unique to the instrumentation and is used to create Spans.
Instrumentation should be designed to accept a TracerProvider from which it
can create its own unique Tracer. Alternatively, the registered global
TracerProvider from the go.opentelemetry.io/otel package can be used as
a default.

	const (
		name    = "instrumentation/package/name"
		version = "0.1.0"
	)

	type Instrumentation struct {
		tracer trace.Tracer
	}

	func NewInstrumentation(tp trace.TracerProvider) *Instrumentation {
		if tp == nil {
			tp = otel.TracerProvider()
		}
		return &Instrumentation{
			tracer: tp.Tracer(name, trace.WithInstrumentationVersion(version)),
		}
	}

	func operation(ctx context.Context, inst *Instrumentation) {
		var span trace.Span
		ctx, span = inst.tracer.Start(ctx, "operation")
		defer span.End()
		// ...
	}
*/
package trace // import "go.opentelemetry.io/otel/trace"
