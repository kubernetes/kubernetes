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

package propagation

import (
	"context"
)

// HTTPSupplier is an interface that specifies methods to retrieve and
// store a single value for a key to an associated carrier. It is
// implemented by http.Headers.
type HTTPSupplier interface {
	// Get method retrieves a single value for a given key.
	Get(key string) string
	// Set method stores a single value for a given key. Note that
	// this should not be appending a value to some array, but
	// rather overwrite the old value.
	Set(key string, value string)
}

// HTTPExtractor extracts information from a HTTPSupplier into a
// context.
type HTTPExtractor interface {
	// Extract method retrieves encoded information using supplier
	// from the associated carrier, decodes it and creates a new
	// context containing the decoded information.
	//
	// Information can be a correlation context or a remote span
	// context. In case of span context, the propagator should
	// store it in the context using
	// trace.ContextWithRemoteSpanContext. In case of correlation
	// context, the propagator should use correlation.WithMap to
	// store it in the context.
	Extract(context.Context, HTTPSupplier) context.Context
}

// HTTPInjector injects information into a HTTPSupplier.
type HTTPInjector interface {
	// Inject method retrieves information from the context,
	// encodes it into propagator specific format and then injects
	// the encoded information using supplier into an associated
	// carrier.
	Inject(context.Context, HTTPSupplier)
}

// Config contains the current set of extractors and injectors.
type Config struct {
	httpEx []HTTPExtractor
	httpIn []HTTPInjector
}

// Propagators is the interface to a set of injectors and extractors
// for all supported carrier formats. It can be used to chain multiple
// propagators into a single entity.
type Propagators interface {
	// HTTPExtractors returns the configured extractors.
	HTTPExtractors() []HTTPExtractor

	// HTTPInjectors returns the configured injectors.
	HTTPInjectors() []HTTPInjector
}

// HTTPPropagator is the interface to inject to and extract from
// HTTPSupplier.
type HTTPPropagator interface {
	HTTPInjector
	HTTPExtractor

	// GetAllKeys returns the HTTP header names used.
	GetAllKeys() []string
}

// Option support passing configuration parameters to New().
type Option func(*Config)

// propagators is the default Propagators implementation.
type propagators struct {
	config Config
}

// New returns a standard Propagators implementation.
func New(options ...Option) Propagators {
	config := Config{}
	for _, opt := range options {
		opt(&config)
	}
	return &propagators{
		config: config,
	}
}

// WithInjectors appends to the optional injector set.
func WithInjectors(inj ...HTTPInjector) Option {
	return func(config *Config) {
		config.httpIn = append(config.httpIn, inj...)
	}
}

// WithExtractors appends to the optional extractor set.
func WithExtractors(ext ...HTTPExtractor) Option {
	return func(config *Config) {
		config.httpEx = append(config.httpEx, ext...)
	}
}

// HTTPExtractors implements Propagators.
func (p *propagators) HTTPExtractors() []HTTPExtractor {
	return p.config.httpEx
}

// HTTPInjectors implements Propagators.
func (p *propagators) HTTPInjectors() []HTTPInjector {
	return p.config.httpIn
}

// ExtractHTTP applies props.HTTPExtractors() to the passed context
// and the supplier and returns the combined result context.
func ExtractHTTP(ctx context.Context, props Propagators, supplier HTTPSupplier) context.Context {
	for _, ex := range props.HTTPExtractors() {
		ctx = ex.Extract(ctx, supplier)
	}
	return ctx
}

// InjectHTTP applies props.HTTPInjectors() to the passed context and
// the supplier.
func InjectHTTP(ctx context.Context, props Propagators, supplier HTTPSupplier) {
	for _, in := range props.HTTPInjectors() {
		in.Inject(ctx, supplier)
	}
}
