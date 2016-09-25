/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	offerSubsystem = "mesos_offers"
)

type OfferDeclinedReason string

const (
	OfferExpired   = OfferDeclinedReason("expired")
	OfferRescinded = OfferDeclinedReason("rescinded")
	OfferCompat    = OfferDeclinedReason("compat")
)

var (
	OffersReceived = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: offerSubsystem,
			Name:      "received",
			Help:      "Counter of offers received from Mesos broken out by slave host.",
		},
		[]string{"hostname"},
	)

	OffersDeclined = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: offerSubsystem,
			Name:      "declined",
			Help:      "Counter of offers declined by the framework broken out by slave host.",
		},
		[]string{"hostname", "reason"},
	)

	OffersAcquired = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: offerSubsystem,
			Name:      "acquired",
			Help:      "Counter of offers acquired for task launch broken out by slave host.",
		},
		[]string{"hostname"},
	)

	OffersReleased = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: offerSubsystem,
			Name:      "released",
			Help:      "Counter of previously-acquired offers later released, broken out by slave host.",
		},
		[]string{"hostname"},
	)
)

var registerMetrics sync.Once

func Register() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(OffersReceived)
		prometheus.MustRegister(OffersDeclined)
		prometheus.MustRegister(OffersAcquired)
		prometheus.MustRegister(OffersReleased)
	})
}

func InMicroseconds(d time.Duration) float64 {
	return float64(d.Nanoseconds() / time.Microsecond.Nanoseconds())
}
