// Copyright 2026 The etcd Authors
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

package read

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	slowReadIndex = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "slow_read_indexes_total",
		Help:      "The total number of pending read indexes not in sync with leader's or timed out read index requests.",
	})
	readIndexFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "read_indexes_failed_total",
		Help:      "The total number of failed read indexes seen.",
	})
)

func init() {
	prometheus.MustRegister(slowReadIndex)
	prometheus.MustRegister(readIndexFailed)
}
