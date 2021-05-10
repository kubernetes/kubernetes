package servicecacertpublisher

import (
	"strconv"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// ServiceCACertPublisher - subsystem name used by service_ca_cert_publisher
const ServiceCACertPublisher = "service_ca_cert_publisher"

var (
	syncCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      ServiceCACertPublisher,
			Name:           "sync_total",
			Help:           "Number of namespace syncs happened in service ca cert publisher.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)
	syncLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      ServiceCACertPublisher,
			Name:           "sync_duration_seconds",
			Help:           "Number of namespace syncs happened in service ca cert publisher.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)
)

func recordMetrics(start time.Time, ns string, err error) {
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
