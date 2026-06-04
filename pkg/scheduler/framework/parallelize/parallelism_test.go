/*
Copyright 2020 The Kubernetes Authors.

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

package parallelize

import (
	"fmt"
	"sync"
	"testing"

	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

func TestChunkSize(t *testing.T) {
	tests := []struct {
		input      int
		wantOutput int
	}{
		{
			input:      32,
			wantOutput: 3,
		},
		{
			input:      16,
			wantOutput: 2,
		},
		{
			input:      1,
			wantOutput: 1,
		},
		{
			input:      0,
			wantOutput: 1,
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%d", test.input), func(t *testing.T) {
			if chunkSizeFor(test.input, DefaultParallelism) != test.wantOutput {
				t.Errorf("Expected: %d, got: %d", test.wantOutput, chunkSizeFor(test.input, DefaultParallelism))
			}
		})
	}
}

func TestGoroutinesMetric(t *testing.T) {
	metrics.Register()
	metrics.Goroutines.Reset()

	const (
		operation = "test-operation"
		pieces    = 32
	)

	var (
		mu        sync.Mutex
		peakValue float64
	)

	_, ctx := ktesting.NewTestContext(t)
	p := NewParallelizer(DefaultParallelism)
	p.Until(ctx, pieces, func(_ int) {
		val, err := testutil.GetGaugeMetricValue(metrics.Goroutines.WithLabelValues(operation))
		if err != nil {
			t.Fatalf("failed to read goroutines metric inside Until: %v", err)
		}
		mu.Lock()
		if val > peakValue {
			peakValue = val
		}
		mu.Unlock()
	}, operation)

	if peakValue <= 0 {
		t.Errorf("expected goroutines metric to be >0 during Until, peak was %v", peakValue)
	}
}
