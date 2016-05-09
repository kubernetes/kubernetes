// Copyright 2015 CoreOS, Inc.
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

package storage

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	rangeCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "range_total",
			Help:      "Total number of ranges seen by this member.",
		})

	putCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "put_total",
			Help:      "Total number of puts seen by this member.",
		})

	deleteCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "delete_total",
			Help:      "Total number of deletes seen by this member.",
		})

	txnCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "txn_total",
			Help:      "Total number of txns seen by this member.",
		})

	keysGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "keys_total",
			Help:      "Total number of keys.",
		})

	watchStreamGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "watch_stream_total",
			Help:      "Total number of watch streams.",
		})

	watcherGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "watcher_total",
			Help:      "Total number of watchers.",
		})

	slowWatcherGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "slow_watcher_total",
			Help:      "Total number of unsynced slow watchers.",
		})

	totalEventsCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "events_total",
			Help:      "Total number of events sent by this member.",
		})

	pendingEventsGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "pending_events_total",
			Help:      "Total number of pending events to be sent.",
		})

	indexCompactionPauseDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "index_compaction_pause_duration_milliseconds",
			Help:      "Bucketed histogram of index compaction pause duration.",
			// 0.5ms -> 1second
			Buckets: prometheus.ExponentialBuckets(0.5, 2, 12),
		})

	dbCompactionPauseDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "db_compaction_pause_duration_milliseconds",
			Help:      "Bucketed histogram of db compaction pause duration.",
			// 1ms -> 4second
			Buckets: prometheus.ExponentialBuckets(1, 2, 13),
		})

	dbCompactionTotalDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "storage",
			Name:      "db_compaction_total_duration_milliseconds",
			Help:      "Bucketed histogram of db compaction total duration.",
			// 100ms -> 800second
			Buckets: prometheus.ExponentialBuckets(100, 2, 14),
		})

	dbTotalSize = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "storage",
		Name:      "db_total_size_in_bytes",
		Help:      "Total size of the underlying database in bytes.",
	})
)

func init() {
	prometheus.MustRegister(rangeCounter)
	prometheus.MustRegister(putCounter)
	prometheus.MustRegister(deleteCounter)
	prometheus.MustRegister(txnCounter)
	prometheus.MustRegister(keysGauge)
	prometheus.MustRegister(watchStreamGauge)
	prometheus.MustRegister(watcherGauge)
	prometheus.MustRegister(slowWatcherGauge)
	prometheus.MustRegister(totalEventsCounter)
	prometheus.MustRegister(pendingEventsGauge)
	prometheus.MustRegister(indexCompactionPauseDurations)
	prometheus.MustRegister(dbCompactionPauseDurations)
	prometheus.MustRegister(dbCompactionTotalDurations)
	prometheus.MustRegister(dbTotalSize)
}

// ReportEventReceived reports that an event is received.
// This function should be called when the external systems received an
// event from storage.Watcher.
func ReportEventReceived() {
	pendingEventsGauge.Dec()
	totalEventsCounter.Inc()
}
