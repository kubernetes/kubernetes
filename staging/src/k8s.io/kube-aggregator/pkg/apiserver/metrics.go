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
	"context"
	"net/http"
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

var (
	extensionApiserverRequestCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "extension_apiserver_requests_total",
			Help:           "Counter of extension apiserver request broken down by result. It can be either 'OK', 'Not Found', 'Service Unavailable' or 'Internal Server Error'.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)

	extensionApiserverLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "extension_apiserver_request_duration_seconds",
			Help:           "extension apiserver request duration in seconds broken out by result.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)
)

func init() {
	legacyregistry.MustRegister(x509MissingSANCounter)
	legacyregistry.MustRegister(x509InsecureSHA1Counter)
	legacyregistry.MustRegister(extensionApiserverRequestCounter)
	legacyregistry.MustRegister(extensionApiserverLatency)
}

func recordExtensionApiserverMetrics(ctx context.Context, httpStatus int, extensionApiserverStart time.Time) {
	extensionApiserverFinish := time.Now()
	extensionApiserverRequestCounter.WithContext(ctx).WithLabelValues(http.StatusText(httpStatus)).Inc()
	extensionApiserverLatency.WithContext(ctx).WithLabelValues(http.StatusText(httpStatus)).Observe(extensionApiserverFinish.Sub(extensionApiserverStart).Seconds())
}
