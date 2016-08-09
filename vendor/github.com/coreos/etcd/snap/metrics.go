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

package snap

import "github.com/prometheus/client_golang/prometheus"

var (
	// TODO: save_fsync latency?
	saveDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd_debugging",
		Subsystem: "snap",
		Name:      "save_total_duration_seconds",
		Help:      "The total latency distributions of save called by snapshot.",
		Buckets:   prometheus.ExponentialBuckets(0.001, 2, 14),
	})

	marshallingDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd_debugging",
		Subsystem: "snap",
		Name:      "save_marshalling_duration_seconds",
		Help:      "The marshalling cost distributions of save called by snapshot.",
		Buckets:   prometheus.ExponentialBuckets(0.001, 2, 14),
	})
)

func init() {
	prometheus.MustRegister(saveDurations)
	prometheus.MustRegister(marshallingDurations)
}
