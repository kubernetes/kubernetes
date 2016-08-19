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
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/clock"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	GarbageCollectSubsystem    = "garbage_collector"
	EventProcessingLatencyKey  = "event_processing_latency_microseconds"
	DirtyProcessingLatencyKey  = "dirty_processing_latency_microseconds"
	OrphanProcessingLatencyKey = "orphan_processing_latency_microseconds"
	DeletionToWatchLatencyKey  = "deletion_to_watch_latency_microseconds"
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
	DeletionToWatchLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: GarbageCollectSubsystem,
			Name:      DeletionToWatchLatencyKey,
			Help:      "Time in microseconds between GC deletes an item and GC observes the deletion via reflector",
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
		prometheus.MustRegister(DeletionToWatchLatency)
	})
}

func sinceInMicroseconds(clock clock.Clock, start time.Time) float64 {
	return float64(clock.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

func NewDeletionToWatchTracker(limit int) *uidToDeletionTime {
	return &uidToDeletionTime{
		items: make(map[types.UID]time.Time),
		lock:  sync.Mutex{},
		limit: limit,
	}
}

type uidToDeletionTime struct {
	items map[types.UID]time.Time
	lock  sync.Mutex
	limit int
}

// DeletionSent should be called when the deletion request is sent.
func (u *uidToDeletionTime) DeletionSent(uid types.UID, start time.Time) error {
	u.lock.Lock()
	defer u.lock.Unlock()
	if len(u.items) >= u.limit {
		return fmt.Errorf("limit %d is reached", u.limit)
	}
	u.items[uid] = start
	return nil
}

// DeletionObserved should be called when the deletion is observed through the reflector.
func (u *uidToDeletionTime) DeletionObserved(uid types.UID, end time.Time) (float64, error) {
	u.lock.Lock()
	defer u.lock.Unlock()
	start, ok := u.items[uid]
	if !ok {
		return 0, fmt.Errorf("cannot find uid %s", uid)
	}
	delete(u.items, uid)
	return float64(end.Sub(start).Nanoseconds() / time.Microsecond.Nanoseconds()), nil
}
