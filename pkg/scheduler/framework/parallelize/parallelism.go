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
	"context"
	"math"

	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// DefaultParallelism is the default parallelism used in scheduler.
const DefaultParallelism int = 16

// Parallelizer holds the parallelism for scheduler.
type Parallelizer struct {
	parallelism int
}

// NewParallelizer returns an object holding the parallelism.
func NewParallelizer(p int) Parallelizer {
	return Parallelizer{parallelism: p}
}

// chunkSizeFor returns a chunk size for the given number of items to use for
// parallel work. The size aims to produce good CPU utilization.
// returns max(1, min(sqrt(n), n/Parallelism))
func chunkSizeFor(n, parallelism int) int {
	s := int(math.Sqrt(float64(n)))

	if r := n/parallelism + 1; s > r {
		s = r
	} else if s < 1 {
		s = 1
	}
	return s
}

// Until is a wrapper around workqueue.ParallelizeUntil to use in scheduling algorithms.
// A given operation will be a label that is recorded in the goroutine metric.
func (p Parallelizer) Until(ctx context.Context, pieces int, doWorkPiece workqueue.DoWorkPieceFunc, operation string) {
	goroutinesMetric := metrics.Goroutines.WithLabelValues(operation)
	withMetrics := func(piece int) {
		goroutinesMetric.Inc()
		doWorkPiece(piece)
		goroutinesMetric.Dec()
	}

	workqueue.ParallelizeUntil(ctx, p.parallelism, pieces, withMetrics, workqueue.WithChunkSize(chunkSizeFor(pieces, p.parallelism)))
}
