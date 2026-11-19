/*
Copyright The Kubernetes Authors.

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

// Package metrics provides interfaces for instrumenting the CRI client without
// requiring a particular metrics implementation.
package metrics

import (
	"context"
	"sync"
)

// ListStreamingMetric counts streaming attempts by operation and result.
// Result is success, error, or fallback for Unimplemented errors.
type ListStreamingMetric interface {
	Increment(ctx context.Context, operation, result string)
}

var (
	// ListStreamingTotal counts streaming RPC attempts, including calls that fall back to unary.
	ListStreamingTotal ListStreamingMetric = noopListStreaming{}
	registerMetrics    sync.Once
)

// RegisterOpts contains the metrics to register. Nil metrics remain no-ops.
type RegisterOpts struct {
	ListStreamingTotal ListStreamingMetric
}

// Register installs metrics once and must be called before using CRI clients.
func Register(opts RegisterOpts) {
	registerMetrics.Do(func() {
		if opts.ListStreamingTotal != nil {
			ListStreamingTotal = opts.ListStreamingTotal
		}
	})
}

type noopListStreaming struct{}

func (noopListStreaming) Increment(context.Context, string, string) {}
