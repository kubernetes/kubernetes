// Copyright 2016 The etcd Authors
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

package backend

import "github.com/prometheus/client_golang/prometheus"

var (
	commitDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "disk",
		Name:      "backend_commit_duration_seconds",
		Help:      "The latency distributions of commit called by backend.",
		Buckets:   prometheus.ExponentialBuckets(0.001, 2, 14),
	})

	snapshotDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "disk",
		Name:      "backend_snapshot_duration_seconds",
		Help:      "The latency distribution of backend snapshots.",
		// 10 ms -> 655 seconds
		Buckets: prometheus.ExponentialBuckets(.01, 2, 17),
	})
)

func init() {
	prometheus.MustRegister(commitDurations)
	prometheus.MustRegister(snapshotDurations)
}
