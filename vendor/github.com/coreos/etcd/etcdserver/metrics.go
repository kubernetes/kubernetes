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

package etcdserver

import (
	goruntime "runtime"
	"time"

	"github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/version"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	hasLeader = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "has_leader",
		Help:      "Whether or not a leader exists. 1 is existence, 0 is not.",
	})
	isLeader = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "is_leader",
		Help:      "Whether or not this member is a leader. 1 if is, 0 otherwise.",
	})
	leaderChanges = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "leader_changes_seen_total",
		Help:      "The number of leader changes seen.",
	})
	heartbeatSendFailures = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "heartbeat_send_failures_total",
		Help:      "The total number of leader heartbeat send failures (likely overloaded from slow disk).",
	})
	slowApplies = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "slow_apply_total",
		Help:      "The total number of slow apply requests (likely overloaded from slow disk).",
	})
	proposalsCommitted = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposals_committed_total",
		Help:      "The total number of consensus proposals committed.",
	})
	proposalsApplied = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposals_applied_total",
		Help:      "The total number of consensus proposals applied.",
	})
	proposalsPending = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposals_pending",
		Help:      "The current number of pending proposals to commit.",
	})
	proposalsFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "proposals_failed_total",
		Help:      "The total number of failed proposals seen.",
	})
	leaseExpired = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd_debugging",
		Subsystem: "server",
		Name:      "lease_expired_total",
		Help:      "The total number of expired leases.",
	})
	slowReadIndex = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "slow_read_indexes_total",
		Help:      "The total number of pending read indexes not in sync with leader's or timed out read index requests.",
	})
	quotaBackendBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "quota_backend_bytes",
		Help:      "Current backend storage quota size in bytes.",
	})
	currentVersion = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "version",
		Help:      "Which version is running. 1 for 'server_version' label with current version.",
	},
		[]string{"server_version"})
	currentGoVersion = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "go_version",
		Help:      "Which Go version server is running with. 1 for 'server_go_version' label with current version.",
	},
		[]string{"server_go_version"})
)

func init() {
	prometheus.MustRegister(hasLeader)
	prometheus.MustRegister(isLeader)
	prometheus.MustRegister(leaderChanges)
	prometheus.MustRegister(heartbeatSendFailures)
	prometheus.MustRegister(slowApplies)
	prometheus.MustRegister(proposalsCommitted)
	prometheus.MustRegister(proposalsApplied)
	prometheus.MustRegister(proposalsPending)
	prometheus.MustRegister(proposalsFailed)
	prometheus.MustRegister(leaseExpired)
	prometheus.MustRegister(slowReadIndex)
	prometheus.MustRegister(quotaBackendBytes)
	prometheus.MustRegister(currentVersion)
	prometheus.MustRegister(currentGoVersion)

	currentVersion.With(prometheus.Labels{
		"server_version": version.Version,
	}).Set(1)
	currentGoVersion.With(prometheus.Labels{
		"server_go_version": goruntime.Version(),
	}).Set(1)
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
