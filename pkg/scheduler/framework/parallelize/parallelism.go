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

// Parallelizer implements k8s.io/kube-scheduler/framework.Parallelizer helps run scheduling operations in parallel chunks where possible, to improve performance and CPU utilization.
// It wraps logic of k8s.io/client-go/util/workqueue to run operations on multiple workers.
type Parallelizer struct {
	parallelism int
}

// NewParallelizer returns an object holding the parallelism (number of workers).
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

// numWorkersForChunkSize returns number of workers (goroutines)
// that will be created in workqueue.ParallelizeUntil
// for given parallelism, pieces and chunkSize values.
func numWorkersForChunkSize(parallelism, pieces, chunkSize int) int {
	chunks := (pieces + chunkSize - 1) / chunkSize
	if chunks < parallelism {
		return chunks
	}
	return parallelism
}

// Until is a wrapper around workqueue.ParallelizeUntil to use in scheduling algorithms.
// A given operation will be a label that is recorded in the goroutine metric.
func (p Parallelizer) Until(ctx context.Context, pieces int, doWorkPiece workqueue.DoWorkPieceFunc, operation string) {
	chunkSize := chunkSizeFor(pieces, p.parallelism)
	workers := numWorkersForChunkSize(p.parallelism, pieces, chunkSize)

	goroutinesMetric := metrics.Goroutines.WithLabelValues(operation)
	// Calling single Add with workers' count is more efficient than calling Inc or Dec per each work piece.
	// This approach improves performance of some plugins (affinity, topology spreading) as well as preemption.
	goroutinesMetric.Add(float64(workers))
	defer goroutinesMetric.Add(float64(-workers))

	workqueue.ParallelizeUntil(ctx, p.parallelism, pieces, doWorkPiece, workqueue.WithChunkSize(chunkSize))
}
