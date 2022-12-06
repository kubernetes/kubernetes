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

package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/propagation"
)

// textMapPropagator is a default TextMapPropagator that delegates calls to a
// registered delegate if one is set, otherwise it defaults to delegating the
// calls to a the default no-op propagation.TextMapPropagator.
type textMapPropagator struct {
	mtx      sync.Mutex
	once     sync.Once
	delegate propagation.TextMapPropagator
	noop     propagation.TextMapPropagator
}

// Compile-time guarantee that textMapPropagator implements the
// propagation.TextMapPropagator interface.
var _ propagation.TextMapPropagator = (*textMapPropagator)(nil)

func newTextMapPropagator() *textMapPropagator {
	return &textMapPropagator{
		noop: propagation.NewCompositeTextMapPropagator(),
	}
}

// SetDelegate sets a delegate propagation.TextMapPropagator that all calls are
// forwarded to. Delegation can only be performed once, all subsequent calls
// perform no delegation.
func (p *textMapPropagator) SetDelegate(delegate propagation.TextMapPropagator) {
	if delegate == nil {
		return
	}

	p.mtx.Lock()
	p.once.Do(func() { p.delegate = delegate })
	p.mtx.Unlock()
}

// effectiveDelegate returns the current delegate of p if one is set,
// otherwise the default noop TextMapPropagator is returned. This method
// can be called concurrently.
func (p *textMapPropagator) effectiveDelegate() propagation.TextMapPropagator {
	p.mtx.Lock()
	defer p.mtx.Unlock()
	if p.delegate != nil {
		return p.delegate
	}
	return p.noop
}

// Inject set cross-cutting concerns from the Context into the carrier.
func (p *textMapPropagator) Inject(ctx context.Context, carrier propagation.TextMapCarrier) {
	p.effectiveDelegate().Inject(ctx, carrier)
}

// Extract reads cross-cutting concerns from the carrier into a Context.
func (p *textMapPropagator) Extract(ctx context.Context, carrier propagation.TextMapCarrier) context.Context {
	return p.effectiveDelegate().Extract(ctx, carrier)
}

// Fields returns the keys whose values are set with Inject.
func (p *textMapPropagator) Fields() []string {
	return p.effectiveDelegate().Fields()
}
