/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	proxyActiveCounts = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "aggregator_proxy_active",
			Help: "Indicates if the proxy is active for a label (group/version).",
		},
		[]string{"apiservice"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(proxyActiveCounts)
	})
}

func SetProxyingGroupVersionActive(apiServiceName string) {
	proxyActiveCounts.WithLabelValues(apiServiceName).Set(float64(1))
}
func SetProxyingGroupVersionInactive(apiServiceName string) {
	proxyActiveCounts.WithLabelValues(apiServiceName).Set(float64(0))
}
