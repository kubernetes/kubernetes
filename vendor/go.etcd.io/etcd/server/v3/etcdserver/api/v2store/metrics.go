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

package v2store

import "github.com/prometheus/client_golang/prometheus"

// Set of raw Prometheus metrics.
// Labels
// * action = declared in event.go
// * outcome = Outcome
// Do not increment directly, use Report* methods.
var (
	readCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "reads_total",
			Help:      "Total number of reads action by (get/getRecursive), local to this member.",
		},
		[]string{"action"},
	)

	writeCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "writes_total",
			Help:      "Total number of writes (e.g. set/compareAndDelete) seen by this member.",
		},
		[]string{"action"},
	)

	readFailedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "reads_failed_total",
			Help:      "Failed read actions by (get/getRecursive), local to this member.",
		},
		[]string{"action"},
	)

	writeFailedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "writes_failed_total",
			Help:      "Failed write actions (e.g. set/compareAndDelete), seen by this member.",
		},
		[]string{"action"},
	)

	expireCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "expires_total",
			Help:      "Total number of expired keys.",
		},
	)

	watchRequests = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "watch_requests_total",
			Help:      "Total number of incoming watch requests (new or reestablished).",
		},
	)

	watcherCount = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "etcd_debugging",
			Subsystem: "store",
			Name:      "watchers",
			Help:      "Count of currently active watchers.",
		},
	)
)

const (
	GetRecursive = "getRecursive"
)

func init() {
	if prometheus.Register(readCounter) != nil {
		// Tests will try to double register since the tests use both
		// store and store_test packages; ignore second attempts.
		return
	}
	prometheus.MustRegister(writeCounter)
	prometheus.MustRegister(expireCounter)
	prometheus.MustRegister(watchRequests)
	prometheus.MustRegister(watcherCount)
}

func reportReadSuccess(readAction string) {
	readCounter.WithLabelValues(readAction).Inc()
}

func reportReadFailure(readAction string) {
	readCounter.WithLabelValues(readAction).Inc()
	readFailedCounter.WithLabelValues(readAction).Inc()
}

func reportWriteSuccess(writeAction string) {
	writeCounter.WithLabelValues(writeAction).Inc()
}

func reportWriteFailure(writeAction string) {
	writeCounter.WithLabelValues(writeAction).Inc()
	writeFailedCounter.WithLabelValues(writeAction).Inc()
}

func reportExpiredKey() {
	expireCounter.Inc()
}

func reportWatchRequest() {
	watchRequests.Inc()
}

func reportWatcherAdded() {
	watcherCount.Inc()
}

func reportWatcherRemoved() {
	watcherCount.Dec()
}
