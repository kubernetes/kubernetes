/*
Copyright 2023 The Kubernetes Authors.

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

package ktesting

import (
	"testing"
	"time"
)

// RunB runs f as a subbenchmark of tCtx called name.
// tCtx must be based on *testing.B, otherwise
// RunB fails the parent benchmark.
func RunB(tCtx TContext, name string, f func(bCtx BContext)) {
	tb := tCtx.TB()
	b, ok := tb.(*testing.B)
	if !ok {
		tCtx.Helper()
		tCtx.Errorf("ktesting.RunB is only supported for a TContext based on *testing.B, got instead: %T", tb)
	}
	b.Run(name, func(b *testing.B) {
		tCtx := WithTB(tCtx, b)
		f(BContext{N: b.N, TContext: tCtx, b: b})
	})
}

// BContext is a variant of TContext which exposes the additional API normally
// associated with a testing.B (N field, methods for controlling benchmarks).
type BContext struct {
	TContext
	N int
	b *testing.B
}

func (bCtx BContext) Elapsed() time.Duration {
	return bCtx.b.Elapsed()
}

func (bCtx BContext) ReportAllocs() {
	bCtx.b.ReportAllocs()
}

func (bCtx BContext) ReportMetric(n float64, unit string) {
	bCtx.b.ReportMetric(n, unit)
}

func (bCtx BContext) ResetTimer() {
	bCtx.b.ResetTimer()
}

func (bCtx BContext) SetBytes(n int64) {
	bCtx.b.SetBytes(n)
}

func (bCtx BContext) StartTimer() {
	bCtx.b.StartTimer()
}

func (bCtx BContext) StopTimer() {
	bCtx.b.StopTimer()
}

// RunT runs f as a subtest of tCtx called name.
// tCtx must be based on *testing.T, otherwise
// RunT fails the parent test.
func RunT(tCtx TContext, name string, f func(tCtx TContext)) {
	tb := tCtx.TB()
	t, ok := tb.(*testing.T)
	if !ok {
		tCtx.Helper()
		tCtx.Errorf("ktesting.RunT is only supported for a TContext based on *testing.T, got instead: %T", tb)
	}
	t.Run(name, func(t *testing.T) {
		tCtx := WithTB(tCtx, t)
		f(tCtx)
	})
}
