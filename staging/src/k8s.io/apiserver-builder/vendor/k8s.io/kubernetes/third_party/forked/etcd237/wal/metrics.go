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

package wal

import "github.com/prometheus/client_golang/prometheus"

var (
	syncDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "wal",
		Name:      "fsync_durations_seconds",
		Help:      "The latency distributions of fsync called by wal.",
		Buckets:   prometheus.ExponentialBuckets(0.001, 2, 14),
	})
	lastIndexSaved = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "wal",
		Name:      "last_index_saved",
		Help:      "The index of the last entry saved by wal.",
	})
)

func init() {
	prometheus.MustRegister(syncDurations)
	prometheus.MustRegister(lastIndexSaved)
}
