/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	GarbageCollectSubsystem    = "garbage_collector"
	EventProcessingLatencyKey  = "event_processing_latency_microseconds"
	DirtyProcessingLatencyKey  = "dirty_processing_latency_microseconds"
	OrphanProcessingLatencyKey = "orphan_processing_latency_microseconds"
)

var (
	EventProcessingLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: GarbageCollectSubsystem,
			Name:      EventProcessingLatencyKey,
			Help:      "Time in microseconds of an event spend in the eventQueue",
		},
	)
	DirtyProcessingLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: GarbageCollectSubsystem,
			Name:      DirtyProcessingLatencyKey,
			Help:      "Time in microseconds of an item spend in the dirtyQueue",
		},
	)
	OrphanProcessingLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: GarbageCollectSubsystem,
			Name:      OrphanProcessingLatencyKey,
			Help:      "Time in microseconds of an item spend in the orphanQueue",
		},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(EventProcessingLatency)
		prometheus.MustRegister(DirtyProcessingLatency)
		prometheus.MustRegister(OrphanProcessingLatency)
	})
}

func sinceInMicroseconds(clock clock.Clock, start time.Time) float64 {
	return float64(clock.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
