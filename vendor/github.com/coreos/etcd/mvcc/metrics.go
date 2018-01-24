// Copyright 2015 The etcd Authors
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

package mvcc

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	rangeCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "range_total",
			Help:      "Total number of ranges seen by this member.",
		})

	putCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "put_total",
			Help:      "Total number of puts seen by this member.",
		})

	deleteCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "delete_total",
			Help:      "Total number of deletes seen by this member.",
		})

	txnCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "txn_total",
			Help:      "Total number of txns seen by this member.",
		})

	keysGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "keys_total",
			Help:      "Total number of keys.",
		})

	watchStreamGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "watch_stream_total",
			Help:      "Total number of watch streams.",
		})

	watcherGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "watcher_total",
			Help:      "Total number of watchers.",
		})

	slowWatcherGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "slow_watcher_total",
			Help:      "Total number of unsynced slow watchers.",
		})

	totalEventsCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "events_total",
			Help:      "Total number of events sent by this member.",
		})

	pendingEventsGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "pending_events_total",
			Help:      "Total number of pending events to be sent.",
		})

	indexCompactionPauseDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "index_compaction_pause_duration_milliseconds",
			Help:      "Bucketed histogram of index compaction pause duration.",
			// 0.5ms -> 1second
			Buckets: prometheus.ExponentialBuckets(0.5, 2, 12),
		})

	dbCompactionPauseDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "db_compaction_pause_duration_milliseconds",
			Help:      "Bucketed histogram of db compaction pause duration.",
			// 1ms -> 4second
			Buckets: prometheus.ExponentialBuckets(1, 2, 13),
		})

	dbCompactionTotalDurations = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "db_compaction_total_duration_milliseconds",
			Help:      "Bucketed histogram of db compaction total duration.",
			// 100ms -> 800second
			Buckets: prometheus.ExponentialBuckets(100, 2, 14),
		})

	dbCompactionKeysCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "mvcc",
			Name:      "db_compaction_keys_total",
			Help:      "Total number of db keys compacted.",
		})

	dbTotalSize = prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "etcd_debugging",
		Subsystem: "mvcc",
		Name:      "db_total_size_in_bytes",
		Help:      "Total size of the underlying database in bytes.",
	},
		func() float64 {
			reportDbTotalSizeInBytesMu.RLock()
			defer reportDbTotalSizeInBytesMu.RUnlock()
			return reportDbTotalSizeInBytes()
		},
	)
	// overridden by mvcc initialization
	reportDbTotalSizeInBytesMu sync.RWMutex
	reportDbTotalSizeInBytes   func() float64 = func() float64 { return 0 }
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
	prometheus.MustRegister(dbCompactionKeysCounter)
	prometheus.MustRegister(dbTotalSize)
}

// ReportEventReceived reports that an event is received.
// This function should be called when the external systems received an
// event from mvcc.Watcher.
func ReportEventReceived(n int) {
	pendingEventsGauge.Sub(float64(n))
	totalEventsCounter.Add(float64(n))
}
