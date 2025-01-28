/*
Copyright 2024 The Kubernetes Authors.

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

package clustertrustbundlepublisher

import (
	"errors"
	"strconv"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// clustertrustbundlePublisher - subsystem name used by clustertrustbundle_publisher
const (
	clustertrustbundlePublisher = "clustertrustbundle_publisher"
)

var (
	syncCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      clustertrustbundlePublisher,
			Name:           "sync_total",
			Help:           "Number of syncs that occurred in cluster trust bundle publisher.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)
	syncLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      clustertrustbundlePublisher,
			Name:           "sync_duration_seconds",
			Help:           "The time it took to sync a cluster trust bundle.",
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
	} else {
		var statusErr apierrors.APIStatus
		if errors.As(err, &statusErr) && statusErr.Status().Code != 0 {
			code = strconv.Itoa(int(statusErr.Status().Code))
		}
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
