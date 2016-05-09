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

package etcdserver

import (
	"time"

	"github.com/coreos/etcd/pkg/runtime"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	// TODO: with label in v3?
	proposeDurations = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposal_durations_seconds",
		Help:      "The latency distributions of committing proposal.",
		Buckets:   prometheus.ExponentialBuckets(0.001, 2, 14),
	})
	proposePending = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "pending_proposal_total",
		Help:      "The total number of pending proposals.",
	})
	// This is number of proposal failed in client's view.
	// The proposal might be later got committed in raft.
	proposeFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposal_failed_total",
		Help:      "The total number of failed proposals.",
	})

	fileDescriptorUsed = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "file_descriptors_used_total",
		Help:      "The total number of file descriptors used.",
	})
)

func init() {
	prometheus.MustRegister(proposeDurations)
	prometheus.MustRegister(proposePending)
	prometheus.MustRegister(proposeFailed)
	prometheus.MustRegister(fileDescriptorUsed)
}

func monitorFileDescriptor(done <-chan struct{}) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		used, err := runtime.FDUsage()
		if err != nil {
			plog.Errorf("cannot monitor file descriptor usage (%v)", err)
			return
		}
		fileDescriptorUsed.Set(float64(used))
		limit, err := runtime.FDLimit()
		if err != nil {
			plog.Errorf("cannot monitor file descriptor usage (%v)", err)
			return
		}
		if used >= limit/5*4 {
			plog.Warningf("80%% of the file descriptor limit is used [used = %d, limit = %d]", used, limit)
		}
		select {
		case <-ticker.C:
		case <-done:
			return
		}
	}
}
