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

package trace

import (
	"context"

	apitrace "go.opentelemetry.io/otel/api/trace"
	"go.opentelemetry.io/otel/internal/trace/parent"
)

type tracer struct {
	provider *Provider
	name     string
}

var _ apitrace.Tracer = &tracer{}

func (tr *tracer) Start(ctx context.Context, name string, o ...apitrace.StartOption) (context.Context, apitrace.Span) {
	var opts apitrace.StartConfig

	for _, op := range o {
		op(&opts)
	}

	parentSpanContext, remoteParent, links := parent.GetSpanContextAndLinks(ctx, opts.NewRoot)

	if p := apitrace.SpanFromContext(ctx); p != nil {
		if sdkSpan, ok := p.(*span); ok {
			sdkSpan.addChild()
		}
	}

	span := startSpanInternal(tr, name, parentSpanContext, remoteParent, opts)
	for _, l := range links {
		span.addLink(l)
	}
	for _, l := range opts.Links {
		span.addLink(l)
	}
	span.SetAttributes(opts.Attributes...)

	span.tracer = tr

	if span.IsRecording() {
		sps, _ := tr.provider.spanProcessors.Load().(spanProcessorMap)
		for sp := range sps {
			sp.OnStart(span.data)
		}
	}

	ctx, end := startExecutionTracerTask(ctx, name)
	span.executionTracerTaskEnd = end
	return apitrace.ContextWithSpan(ctx, span), span
}

func (tr *tracer) WithSpan(ctx context.Context, name string, body func(ctx context.Context) error, opts ...apitrace.StartOption) error {
	ctx, span := tr.Start(ctx, name, opts...)
	defer span.End()

	if err := body(ctx); err != nil {
		// TODO: set event with boolean attribute for error.
		return err
	}
	return nil
}
