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

package rootcacertpublisher

import (
	"strconv"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// RootCACertPublisher - subsystem name used by root_ca_cert_publisher
const RootCACertPublisher = "root_ca_cert_publisher"

var (
	syncCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      RootCACertPublisher,
			Name:           "sync_total",
			Help:           "Number of namespace syncs happened in root ca cert publisher.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)
	syncLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      RootCACertPublisher,
			Name:           "sync_duration_seconds",
			Help:           "Number of namespace syncs happened in root ca cert publisher.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)
)

func recordMetrics(start time.Time, err error) {
	code := "500"
	if err == nil {
		code = "200"
	} else if se, ok := err.(*apierrors.StatusError); ok && se.Status().Code != 0 {
		code = strconv.Itoa(int(se.Status().Code))
	}
	syncLatency.WithLabelValues(code).Observe(time.Since(start).Seconds())
	syncCounter.WithLabelValues(code).Inc()
}

var once sync.Once

func registerMetrics() {
	once.Do(func() {
		legacyregistry.MustRegister(syncCounter)
		legacyregistry.MustRegister(syncLatency)
	})
}
