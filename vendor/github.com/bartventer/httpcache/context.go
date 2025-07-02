// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package httpcache

import (
	"context"

	"github.com/bartventer/httpcache/internal"
)

// ContextWithTraceID adds a trace ID to the context, which can be used for
// logging or tracing purposes. The trace ID can be retrieved later using
// [TraceIDFromContext].
func ContextWithTraceID(ctx context.Context, traceID string) context.Context {
	return context.WithValue(ctx, internal.TraceIDKey, traceID)
}

// TraceIDFromContext retrieves the trace ID from the context, if it exists.
func TraceIDFromContext(ctx context.Context) (string, bool) {
	return internal.TraceIDFromContext(ctx)
}
