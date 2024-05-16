// Copyright 2017, OpenCensus Authors
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
Package trace contains support for OpenCensus distributed tracing.

The following assumes a basic familiarity with OpenCensus concepts.
See http://opencensus.io

# Exporting Traces

To export collected tracing data, register at least one exporter. You can use
one of the provided exporters or write your own.

	trace.RegisterExporter(exporter)

By default, traces will be sampled relatively rarely. To change the sampling
frequency for your entire program, call ApplyConfig. Use a ProbabilitySampler
to sample a subset of traces, or use AlwaysSample to collect a trace on every run:

	trace.ApplyConfig(trace.Config{DefaultSampler: trace.AlwaysSample()})

Be careful about using trace.AlwaysSample in a production application with
significant traffic: a new trace will be started and exported for every request.

# Adding Spans to a Trace

A trace consists of a tree of spans. In Go, the current span is carried in a
context.Context.

It is common to want to capture all the activity of a function call in a span. For
this to work, the function must take a context.Context as a parameter. Add these two
lines to the top of the function:

	ctx, span := trace.StartSpan(ctx, "example.com/Run")
	defer span.End()

StartSpan will create a new top-level span if the context
doesn't contain another span, otherwise it will create a child span.
*/
package trace // import "go.opencensus.io/trace"
