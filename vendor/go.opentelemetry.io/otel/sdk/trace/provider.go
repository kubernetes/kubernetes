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

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/otel/sdk/instrumentation"
	"go.opentelemetry.io/otel/sdk/resource"
)

const (
	defaultTracerName = "go.opentelemetry.io/otel/sdk/tracer"
)

// TODO (MrAlias): unify this API option design:
// https://github.com/open-telemetry/opentelemetry-go/issues/536

// TracerProviderConfig
type TracerProviderConfig struct {
	processors []SpanProcessor

	// sampler is the default sampler used when creating new spans.
	sampler Sampler

	// idGenerator is used to generate all Span and Trace IDs when needed.
	idGenerator IDGenerator

	// spanLimits defines the attribute, event, and link limits for spans.
	spanLimits SpanLimits

	// resource contains attributes representing an entity that produces telemetry.
	resource *resource.Resource
}

type TracerProviderOption func(*TracerProviderConfig)

type TracerProvider struct {
	mu             sync.Mutex
	namedTracer    map[instrumentation.Library]*tracer
	spanProcessors atomic.Value
	sampler        Sampler
	idGenerator    IDGenerator
	spanLimits     SpanLimits
	resource       *resource.Resource
}

var _ trace.TracerProvider = &TracerProvider{}

// NewTracerProvider returns a new and configured TracerProvider.
//
// By default the returned TracerProvider is configured with:
//  - a ParentBased(AlwaysSample) Sampler
//  - a random number IDGenerator
//  - the resource.Default() Resource
//  - the default SpanLimits.
//
// The passed opts are used to override these default values and configure the
// returned TracerProvider appropriately.
func NewTracerProvider(opts ...TracerProviderOption) *TracerProvider {
	o := &TracerProviderConfig{}

	for _, opt := range opts {
		opt(o)
	}

	ensureValidTracerProviderConfig(o)

	tp := &TracerProvider{
		namedTracer: make(map[instrumentation.Library]*tracer),
		sampler:     o.sampler,
		idGenerator: o.idGenerator,
		spanLimits:  o.spanLimits,
		resource:    o.resource,
	}

	for _, sp := range o.processors {
		tp.RegisterSpanProcessor(sp)
	}

	return tp
}

// Tracer returns a Tracer with the given name and options. If a Tracer for
// the given name and options does not exist it is created, otherwise the
// existing Tracer is returned.
//
// If name is empty, DefaultTracerName is used instead.
//
// This method is safe to be called concurrently.
func (p *TracerProvider) Tracer(name string, opts ...trace.TracerOption) trace.Tracer {
	c := trace.NewTracerConfig(opts...)

	p.mu.Lock()
	defer p.mu.Unlock()
	if name == "" {
		name = defaultTracerName
	}
	il := instrumentation.Library{
		Name:    name,
		Version: c.InstrumentationVersion,
	}
	t, ok := p.namedTracer[il]
	if !ok {
		t = &tracer{
			provider:               p,
			instrumentationLibrary: il,
		}
		p.namedTracer[il] = t
	}
	return t
}

// RegisterSpanProcessor adds the given SpanProcessor to the list of SpanProcessors
func (p *TracerProvider) RegisterSpanProcessor(s SpanProcessor) {
	p.mu.Lock()
	defer p.mu.Unlock()
	new := spanProcessorStates{}
	if old, ok := p.spanProcessors.Load().(spanProcessorStates); ok {
		new = append(new, old...)
	}
	newSpanSync := &spanProcessorState{
		sp:    s,
		state: &sync.Once{},
	}
	new = append(new, newSpanSync)
	p.spanProcessors.Store(new)
}

// UnregisterSpanProcessor removes the given SpanProcessor from the list of SpanProcessors
func (p *TracerProvider) UnregisterSpanProcessor(s SpanProcessor) {
	p.mu.Lock()
	defer p.mu.Unlock()
	spss := spanProcessorStates{}
	old, ok := p.spanProcessors.Load().(spanProcessorStates)
	if !ok || len(old) == 0 {
		return
	}
	spss = append(spss, old...)

	// stop the span processor if it is started and remove it from the list
	var stopOnce *spanProcessorState
	var idx int
	for i, sps := range spss {
		if sps.sp == s {
			stopOnce = sps
			idx = i
		}
	}
	if stopOnce != nil {
		stopOnce.state.Do(func() {
			if err := s.Shutdown(context.Background()); err != nil {
				otel.Handle(err)
			}
		})
	}
	if len(spss) > 1 {
		copy(spss[idx:], spss[idx+1:])
	}
	spss[len(spss)-1] = nil
	spss = spss[:len(spss)-1]

	p.spanProcessors.Store(spss)
}

// ForceFlush immediately exports all spans that have not yet been exported for
// all the registered span processors.
func (p *TracerProvider) ForceFlush(ctx context.Context) error {
	spss, ok := p.spanProcessors.Load().(spanProcessorStates)
	if !ok {
		return fmt.Errorf("failed to load span processors")
	}
	if len(spss) == 0 {
		return nil
	}

	for _, sps := range spss {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := sps.sp.ForceFlush(ctx); err != nil {
			return err
		}
	}
	return nil
}

// Shutdown shuts down the span processors in the order they were registered.
func (p *TracerProvider) Shutdown(ctx context.Context) error {
	spss, ok := p.spanProcessors.Load().(spanProcessorStates)
	if !ok {
		return fmt.Errorf("failed to load span processors")
	}
	if len(spss) == 0 {
		return nil
	}

	for _, sps := range spss {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		var err error
		sps.state.Do(func() {
			err = sps.sp.Shutdown(ctx)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// WithSyncer registers the exporter with the TracerProvider using a
// SimpleSpanProcessor.
func WithSyncer(e SpanExporter) TracerProviderOption {
	return WithSpanProcessor(NewSimpleSpanProcessor(e))
}

// WithBatcher registers the exporter with the TracerProvider using a
// BatchSpanProcessor configured with the passed opts.
func WithBatcher(e SpanExporter, opts ...BatchSpanProcessorOption) TracerProviderOption {
	return WithSpanProcessor(NewBatchSpanProcessor(e, opts...))
}

// WithSpanProcessor registers the SpanProcessor with a TracerProvider.
func WithSpanProcessor(sp SpanProcessor) TracerProviderOption {
	return func(opts *TracerProviderConfig) {
		opts.processors = append(opts.processors, sp)
	}
}

// WithResource returns a TracerProviderOption that will configure the
// Resource r as a TracerProvider's Resource. The configured Resource is
// referenced by all the Tracers the TracerProvider creates. It represents the
// entity producing telemetry.
//
// If this option is not used, the TracerProvider will use the
// resource.Default() Resource by default.
func WithResource(r *resource.Resource) TracerProviderOption {
	return func(opts *TracerProviderConfig) {
		opts.resource = resource.Merge(resource.Environment(), r)
	}
}

// WithIDGenerator returns a TracerProviderOption that will configure the
// IDGenerator g as a TracerProvider's IDGenerator. The configured IDGenerator
// is used by the Tracers the TracerProvider creates to generate new Span and
// Trace IDs.
//
// If this option is not used, the TracerProvider will use a random number
// IDGenerator by default.
func WithIDGenerator(g IDGenerator) TracerProviderOption {
	return func(opts *TracerProviderConfig) {
		if g != nil {
			opts.idGenerator = g
		}
	}
}

// WithSampler returns a TracerProviderOption that will configure the Sampler
// s as a TracerProvider's Sampler. The configured Sampler is used by the
// Tracers the TracerProvider creates to make their sampling decisions for the
// Spans they create.
//
// If this option is not used, the TracerProvider will use a
// ParentBased(AlwaysSample) Sampler by default.
func WithSampler(s Sampler) TracerProviderOption {
	return func(opts *TracerProviderConfig) {
		if s != nil {
			opts.sampler = s
		}
	}
}

// WithSpanLimits returns a TracerProviderOption that will configure the
// SpanLimits sl as a TracerProvider's SpanLimits. The configured SpanLimits
// are used used by the Tracers the TracerProvider and the Spans they create
// to limit tracing resources used.
//
// If this option is not used, the TracerProvider will use the default
// SpanLimits.
func WithSpanLimits(sl SpanLimits) TracerProviderOption {
	return func(opts *TracerProviderConfig) {
		opts.spanLimits = sl
	}
}

// ensureValidTracerProviderConfig ensures that given TracerProviderConfig is valid.
func ensureValidTracerProviderConfig(cfg *TracerProviderConfig) {
	if cfg.sampler == nil {
		cfg.sampler = ParentBased(AlwaysSample())
	}
	if cfg.idGenerator == nil {
		cfg.idGenerator = defaultIDGenerator()
	}
	cfg.spanLimits.ensureDefault()
	if cfg.resource == nil {
		cfg.resource = resource.Default()
	}
}
