// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	prometheusPort      = os.Getenv("PROMETHEUS_PORT")
	prometheusPath      = os.Getenv("PROMETHEUS_PATH")
	prometheusNamespace = os.Getenv("PROMETHEUS_NAMESPACE")
	prometheusSubsystem = os.Getenv("PROMETHEUS_SUBSYSTEM")
)

var (
	promDnssecOkCount        prometheus.Counter
	promExternalRequestCount *prometheus.CounterVec
	promRequestCount         *prometheus.CounterVec
	promErrorCount           *prometheus.CounterVec
	promCacheSize            *prometheus.GaugeVec
	promCacheMiss            *prometheus.CounterVec
)

func Metrics() {
	if prometheusPath == "" {
		prometheusPath = "/metrics"
	}
	if prometheusSubsystem == "" {
		prometheusSubsystem = "skydns"
	}

	promExternalRequestCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "dns_request_external_count",
		Help:      "Counter of external DNS requests.",
	}, []string{"type"}) // recursive, stub, lookup
	prometheus.MustRegister(promExternalRequestCount)

	promRequestCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "dns_request_count",
		Help:      "Counter of DNS requests made.",
	}, []string{"type"}) // udp, tcp
	prometheus.MustRegister(promRequestCount)

	promDnssecOkCount = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "dns_dnssec_ok_count",
		Help:      "Counter of DNSSEC requests.",
	})
	prometheus.MustRegister(promDnssecOkCount) // Maybe more bits here?

	promErrorCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "dns_error_count",
		Help:      "Counter of DNS requests resulting in an error.",
	}, []string{"error"}) // nxdomain, nodata, truncated, refused
	prometheus.MustRegister(promErrorCount)

	// Caches
	promCacheSize = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "cache_total_size",
		Help:      "The total size of all elements in the cache.",
	}, []string{"type"}) // response, signature
	prometheus.MustRegister(promCacheSize)

	promCacheMiss = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: prometheusNamespace,
		Subsystem: prometheusSubsystem,
		Name:      "dns_cache_miss_count",
		Help:      "Counter of DNS requests that result in a cache miss.",
	}, []string{"type"}) // response, signature
	prometheus.MustRegister(promCacheMiss)

	if prometheusPort == "" {
		return
	}

	_, err := strconv.Atoi(prometheusPort)
	if err != nil {
		log.Fatalf("skydns: bad port for prometheus: %s", prometheusPort)
	}

	http.Handle(prometheusPath, prometheus.Handler())
	go func() {
		log.Fatalf("skydns: %s", http.ListenAndServe(":"+prometheusPort, nil))
	}()
	log.Printf("skydns: metrics enabled on :%s%s", prometheusPort, prometheusPath)
}

// Counter is the metric interface used by this package
type Counter interface {
	Inc(i int64)
}

type nopCounter struct{}

func (nopCounter) Inc(_ int64) {}

// These are the old stat variables defined by this package. This
// used by graphite.
var (
	// Pondering deletion in favor of the better and more
	// maintained (by me) prometheus reporting.

	StatsForwardCount     Counter = nopCounter{}
	StatsStubForwardCount Counter = nopCounter{}
	StatsLookupCount      Counter = nopCounter{}
	StatsRequestCount     Counter = nopCounter{}
	StatsDnssecOkCount    Counter = nopCounter{}
	StatsNameErrorCount   Counter = nopCounter{}
	StatsNoDataCount      Counter = nopCounter{}

	StatsDnssecCacheMiss Counter = nopCounter{}
)
