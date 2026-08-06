/*
Copyright 2020 The Kubernetes Authors.

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

package apiserver

import (
	"strconv"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var x509MissingSANCounter = metrics.NewCounter(
	&metrics.CounterOpts{
		Subsystem: "kube_aggregator",
		Namespace: "apiserver",
		Name:      "x509_missing_san_total",
		Help: "Counts the number of requests to servers missing SAN extension " +
			"in their serving certificate OR the number of connection failures " +
			"due to the lack of x509 certificate SAN extension missing " +
			"(either/or, based on the runtime environment)",
		StabilityLevel: metrics.ALPHA,
	},
)

var x509InsecureSHA1Counter = metrics.NewCounter(
	&metrics.CounterOpts{
		Subsystem: "kube_aggregator",
		Namespace: "apiserver",
		Name:      "x509_insecure_sha1_total",
		Help: "Counts the number of requests to servers with insecure SHA1 signatures " +
			"in their serving certificate OR the number of connection failures " +
			"due to the insecure SHA1 signatures (either/or, based on the runtime environment)",
		StabilityLevel: metrics.ALPHA,
	},
)

var aggregatorRequestCounter = metrics.NewCounterVec(
	&metrics.CounterOpts{
		Subsystem:      "kube_aggregator",
		Namespace:      "apiserver",
		Name:           "request_total",
		Help:           "Counter of requests proxied by the kube-aggregator to extension API servers, broken down by verb, group, version, and HTTP response code.",
		StabilityLevel: metrics.ALPHA,
	},
	[]string{"verb", "group", "version", "code"},
)

var aggregatorRequestDuration = metrics.NewHistogramVec(
	&metrics.HistogramOpts{
		Subsystem:      "kube_aggregator",
		Namespace:      "apiserver",
		Name:           "request_duration_seconds",
		Help:           "Request duration in seconds for requests proxied by the kube-aggregator to extension API servers, broken down by verb, group, version, and HTTP response code.",
		Buckets:        []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 60},
		StabilityLevel: metrics.ALPHA,
	},
	[]string{"verb", "group", "version", "code"},
)

func init() {
	legacyregistry.MustRegister(x509MissingSANCounter)
	legacyregistry.MustRegister(x509InsecureSHA1Counter)
	legacyregistry.MustRegister(aggregatorRequestCounter)
	legacyregistry.MustRegister(aggregatorRequestDuration)
}

// recordAggregatorRequest reports a request proxied to an extension API server.
// status is the response status code observed at the kube-aggregator boundary;
// callers must pass a non-zero status (200 by convention if no header was written).
func recordAggregatorRequest(verb, group, version string, status int, latency time.Duration) {
	code := strconv.Itoa(status)
	aggregatorRequestCounter.WithLabelValues(verb, group, version, code).Inc()
	aggregatorRequestDuration.WithLabelValues(verb, group, version, code).Observe(latency.Seconds())
}
