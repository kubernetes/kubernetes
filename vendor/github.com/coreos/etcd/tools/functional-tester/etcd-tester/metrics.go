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

package main

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	caseTotalCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "funcational_tester",
			Name:      "case_total",
			Help:      "Total number of finished test cases",
		},
		[]string{"desc"},
	)

	caseFailedTotalCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "funcational_tester",
			Name:      "case_failed_total",
			Help:      "Total number of failed test cases",
		},
		[]string{"desc"},
	)

	roundTotalCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "funcational_tester",
			Name:      "round_total",
			Help:      "Total number of finished test rounds.",
		})

	roundFailedTotalCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "funcational_tester",
			Name:      "round_failed_total",
			Help:      "Total number of failed test rounds.",
		})
)

func init() {
	prometheus.MustRegister(caseTotalCounter)
	prometheus.MustRegister(caseFailedTotalCounter)
	prometheus.MustRegister(roundTotalCounter)
	prometheus.MustRegister(roundFailedTotalCounter)
}
