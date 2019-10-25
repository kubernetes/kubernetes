/*
Copyright 2016 The Kubernetes Authors.

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

package restclient

import (
	"net/url"
	"time"

	"k8s.io/client-go/tools/metrics"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// requestLatency is a Prometheus Summary metric type partitioned by
	// "verb" and "url" labels. It is used for the rest client latency metrics.
	requestLatency = k8smetrics.NewHistogramVec(
		&k8smetrics.HistogramOpts{
			Name:    "rest_client_request_duration_seconds",
			Help:    "Request latency in seconds. Broken down by verb and URL.",
			Buckets: k8smetrics.ExponentialBuckets(0.001, 2, 10),
		},
		[]string{"verb", "url"},
	)

	// deprecatedRequestLatency is deprecated, please use requestLatency.
	deprecatedRequestLatency = k8smetrics.NewHistogramVec(
		&k8smetrics.HistogramOpts{
			Name:              "rest_client_request_latency_seconds",
			Help:              "Request latency in seconds. Broken down by verb and URL.",
			Buckets:           k8smetrics.ExponentialBuckets(0.001, 2, 10),
			DeprecatedVersion: "1.14.0",
		},
		[]string{"verb", "url"},
	)

	requestResult = k8smetrics.NewCounterVec(
		&k8smetrics.CounterOpts{
			Name: "rest_client_requests_total",
			Help: "Number of HTTP requests, partitioned by status code, method, and host.",
		},
		[]string{"code", "method", "host"},
	)

	certExpiration = k8smetrics.NewGauge(
		&k8smetrics.GaugeOpts{
			Name: "rest_client_certificate_expiration_seconds",
			Help: "Gauge of the lifetime of the client certificate. The value is the date the certificate will expire in seconds since January 1, 1970 UTC.",
		},
	)

	certRotation = k8smetrics.NewHistogram(
		&k8smetrics.HistogramOpts{
			Name: "rest_client_certificate_rotation_age",
			Help: "Count of the number of seconds the last client certificate lived before being rotated.",
			// [1m, 1h, ... exponential x^4 ..., 7.5y]
			Buckets: append([]float64{60}, k8smetrics.ExponentialBuckets(3600, 4, 9)...),
		},
	)
)

func init() {
	legacyregistry.MustRegister(requestLatency)
	legacyregistry.MustRegister(deprecatedRequestLatency)
	legacyregistry.MustRegister(requestResult)
	legacyregistry.MustRegister(certExpiration)
	legacyregistry.MustRegister(certRotation)
	metrics.Register(metrics.RegisterOpts{
		ClientCertExpiration:  &expirationAdapter{m: certExpiration},
		ClientCertRotationAge: &rotationAdapter{m: certRotation},
		RequestLatency:        &latencyAdapter{m: requestLatency, dm: deprecatedRequestLatency},
		RequestResult:         &resultAdapter{requestResult},
	})
}

type latencyAdapter struct {
	m  *k8smetrics.HistogramVec
	dm *k8smetrics.HistogramVec
}

func (l *latencyAdapter) Observe(verb string, u url.URL, latency time.Duration) {
	l.m.WithLabelValues(verb, u.String()).Observe(latency.Seconds())
	l.dm.WithLabelValues(verb, u.String()).Observe(latency.Seconds())
}

type resultAdapter struct {
	m *k8smetrics.CounterVec
}

func (r *resultAdapter) Increment(code, method, host string) {
	r.m.WithLabelValues(code, method, host).Inc()
}

type expirationAdapter struct {
	m *k8smetrics.Gauge
}

func (e *expirationAdapter) Set(t time.Time) {
	e.m.Set(float64(t.Unix()))
}

type rotationAdapter struct {
	m *k8smetrics.Histogram
}

func (r *rotationAdapter) Observe(d time.Duration) {
	r.m.Observe(d.Seconds())
}
