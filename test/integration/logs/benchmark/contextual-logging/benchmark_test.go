/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package contextuallogging

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}

// BenchmarkRecursion measures the overhead of adding calling a function
// recursively with just the depth parameter.
func BenchmarkRecursion(b *testing.B) {
	for depth := 10; depth <= 100000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurse(depth)
			}
		})
	}
}

//go:noinline
func recurse(depth int) {
	if depth == 0 {
		logr.Discard().Info("hello world")
		return
	}
	recurse(depth - 1)
}

// BenchmarkRecursionWithLogger measures the overhead of adding a logr.Logger
// parameter.
func BenchmarkRecursionWithLogger(b *testing.B) {
	logger := logr.Discard()

	for depth := 10; depth <= 100000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurseWithLogger(logger, depth)
			}
		})
	}
}

//go:noinline
func recurseWithLogger(logger logr.Logger, depth int) {
	if depth == 0 {
		return
	}
	recurseWithLogger(logger, depth-1)
}

// BenchmarkRecursionWithContext measures the overhead of adding a context
// parameter.
func BenchmarkRecursionWithContext(b *testing.B) {
	logger := logr.Discard()
	// nolint:logcheck // Intentionally using NewContext unconditionally here.
	ctx := logr.NewContext(context.Background(), logger)

	for depth := 10; depth <= 100000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurseWithContext(ctx, depth)
			}
		})
	}
}

//go:noinline
func recurseWithContext(ctx context.Context, depth int) {
	if depth == 0 {
		return
	}
	recurseWithContext(ctx, depth-1)
}

// BenchmarkRecursionWithLogger measures the overhead of adding a logr.Logger
// parameter and using it once.
func BenchmarkRecursionWithLoggerAndLog(b *testing.B) {
	logger := logr.Discard()

	for depth := 10; depth <= 100000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurseWithLoggerAndLog(logger, depth)
			}
		})
	}
}

//go:noinline
func recurseWithLoggerAndLog(logger logr.Logger, depth int) {
	if depth == 0 {
		logger.Info("hello world")
		return
	}
	recurseWithLoggerAndLog(logger, depth-1)
}

// BenchmarkRecursionWithContext measures the overhead of adding a context
// parameter and using it once to retrieve and call a logger.
func BenchmarkRecursionWithContextAndLog(b *testing.B) {
	logger := logr.Discard()
	// nolint:logcheck // Intentionally using NewContext unconditionally here.
	ctx := logr.NewContext(context.Background(), logger)

	for depth := 10; depth <= 100000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurseWithContextAndLog(ctx, depth)
			}
		})
	}
}

//go:noinline
func recurseWithContextAndLog(ctx context.Context, depth int) {
	if depth == 0 {
		logger := logr.FromContextOrDiscard(ctx)
		logger.Info("hello world")
		return
	}
	recurseWithContextAndLog(ctx, depth-1)
}

// BenchmarkNestedContextWithTimeouts benchmarks how quickly a function can be
// called that creates a new context at each call with context.WithTimeout.
func BenchmarkNestedContextWithTimeouts(b *testing.B) {
	ctx := context.Background()

	for depth := 1; depth <= 10000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				nestedContextWithTimeout(ctx, depth)
			}
		})
	}
}

//go:noinline
func nestedContextWithTimeout(ctx context.Context, depth int) {
	if depth == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, time.Hour)
	defer cancel()
	nestedContextWithTimeout(ctx, depth-1)
}

// BenchmarkNestedContextWithTimeouts benchmarks how quickly a function can be
// called that creates a new context at each call with context.WithTimeout
// and then looks up a logger.
func BenchmarkNestedContextWithTimeoutsAndLookup(b *testing.B) {
	// nolint:logcheck // Intentionally using NewContext unconditionally here.
	ctx := logr.NewContext(context.Background(), logr.Discard())

	for depth := 1; depth <= 10000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				nestedContextWithTimeoutAndLookup(ctx, depth)
			}
		})
	}
}

//go:noinline
func nestedContextWithTimeoutAndLookup(ctx context.Context, depth int) {
	if depth == 0 {
		logr.FromContextOrDiscard(ctx)
		return
	}
	ctx, cancel := context.WithTimeout(ctx, time.Hour)
	defer cancel()
	nestedContextWithTimeoutAndLookup(ctx, depth-1)
}

var logger logr.Logger

// BenchmarkNestedContextWithTimeouts benchmarks how quickly FromContextOrDiscard
// can look up a logger in nested contexts where WithTimeouts is used to
// created those nested contexts.
func BenchmarkLookupWithTimeouts(b *testing.B) {
	for depth := 1; depth <= 10000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			// nolint:logcheck // Intentionally using NewContext unconditionally here.
			ctx := logr.NewContext(context.Background(), logr.Discard())
			for i := 0; i < depth; i++ {
				ctx2, cancel := context.WithTimeout(ctx, time.Hour)
				defer cancel()
				ctx = ctx2
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				logger = logr.FromContextOrDiscard(ctx)
			}
		})
	}
}

type keyT struct{}

var key keyT

// BenchmarkNestedContextWithTimeouts benchmarks how quickly FromContextOrDiscard
// can look up a logger in nested contexts where WithValue is used to
// created those nested contexts.
func BenchmarkLookupWithValues(b *testing.B) {
	for depth := 1; depth <= 10000; depth *= 10 {
		b.Run(fmt.Sprintf("%d", depth), func(b *testing.B) {
			// nolint:logcheck // Intentionally using NewContext unconditionally here.
			ctx := logr.NewContext(context.Background(), logr.Discard())
			for i := 0; i < depth; i++ {
				ctx = context.WithValue(ctx, key, depth)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				logger = logr.FromContextOrDiscard(ctx)
			}
		})
	}
}
