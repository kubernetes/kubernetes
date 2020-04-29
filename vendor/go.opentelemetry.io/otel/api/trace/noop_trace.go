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
)

type NoopTracer struct{}

var _ Tracer = NoopTracer{}

// WithSpan wraps around execution of func with noop span.
func (t NoopTracer) WithSpan(ctx context.Context, name string, body func(context.Context) error, opts ...StartOption) error {
	return body(ctx)
}

// Start starts a noop span.
func (NoopTracer) Start(ctx context.Context, name string, opts ...StartOption) (context.Context, Span) {
	span := NoopSpan{}
	return ContextWithSpan(ctx, span), span
}
