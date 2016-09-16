// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package metrics

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/miekg/dns"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	Port      = os.Getenv("PROMETHEUS_PORT")
	Path      = envOrDefault("PROMETHEUS_PATH", "/metrics")
	Namespace = envOrDefault("PROMETHEUS_NAMESPACE", "skydns")
	Subsystem = envOrDefault("PROMETHEUS_SUBSYSTEM", "skydns")

	requestCount    *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec
	responseSize    *prometheus.HistogramVec
	errorCount      *prometheus.CounterVec
	cacheMiss       *prometheus.CounterVec
)

type (
	System    string
	Cause     string
	CacheType string
)

var (
	Auth    System = "auth"
	Cache   System = "cache"
	Rec     System = "recursive"
	Reverse System = "reverse"
	Stub    System = "stub"

	Nxdomain  Cause = "nxdomain"
	Nodata    Cause = "nodata"
	Truncated Cause = "truncated"
	Refused   Cause = "refused"
	Overflow  Cause = "overflow"
	Fail      Cause = "servfail"

	Response  CacheType = "response"
	Signature CacheType = "signature"
)

func defineMetrics() {
	requestCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "dns_request_count_total",
		Help:      "Counter of DNS requests made.",
	}, []string{"system"})

	requestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "dns_request_duration_seconds",
		Help:      "Histogram of the time (in seconds) each request took to resolve.",
		Buckets:   append([]float64{0.001, 0.003}, prometheus.DefBuckets...),
	}, []string{"system"})

	responseSize = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "dns_response_size_bytes",
		Help:      "Size of the returns response in bytes.",
		Buckets: []float64{0, 512, 1024, 1500, 2048, 4096,
			8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864,
			40960, 45056, 49152, 53248, 57344, 61440, 65536,
		},
	}, []string{"system"})

	errorCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "dns_error_count_total",
		Help:      "Counter of DNS requests resulting in an error.",
	}, []string{"system", "cause"})

	cacheMiss = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "dns_cachemiss_count_total",
		Help:      "Counter of DNS requests that result in a cache miss.",
	}, []string{"cache"})
}

// Metrics registers the DNS metrics to Prometheus, and starts the internal metrics
// server if the environment variable PROMETHEUS_PORT is set.
func Metrics() error {
	// We do this in a function instead of using var + init(), because we want to
	// able to set Namespace and/or Subsystem.
	if Port == "" {
		return nil
	}

	_, err := strconv.Atoi(Port)
	if err != nil {
		fmt.Errorf("bad port for prometheus: %s", Port)
	}

	defineMetrics()

	prometheus.MustRegister(requestCount)
	prometheus.MustRegister(requestDuration)
	prometheus.MustRegister(responseSize)
	prometheus.MustRegister(errorCount)
	prometheus.MustRegister(cacheMiss)

	http.Handle(Path, prometheus.Handler())
	go func() {
		fmt.Errorf("%s", http.ListenAndServe(":"+Port, nil))
	}()
	return nil
}

func ReportDuration(resp *dns.Msg, start time.Time, sys System) {
	if requestDuration == nil || responseSize == nil {
		return
	}

	rlen := float64(0)
	if resp != nil {
		rlen = float64(resp.Len())
	}
	requestDuration.WithLabelValues(string(sys)).Observe(float64(time.Since(start)) / float64(time.Second))
	responseSize.WithLabelValues(string(sys)).Observe(rlen)
}

func ReportRequestCount(req *dns.Msg, sys System) {
	if requestCount == nil {
		return
	}

	requestCount.WithLabelValues(string(sys)).Inc()
}

func ReportErrorCount(resp *dns.Msg, sys System) {
	if resp == nil || errorCount == nil {
		return
	}

	if resp.Truncated {
		errorCount.WithLabelValues(string(sys), string(Truncated)).Inc()
		return
	}
	if resp.Len() > dns.MaxMsgSize {
		errorCount.WithLabelValues(string(sys), string(Overflow)).Inc()
		return
	}

	switch resp.Rcode {
	case dns.RcodeServerFailure:
		errorCount.WithLabelValues(string(sys), string(Fail)).Inc()
	case dns.RcodeRefused:
		errorCount.WithLabelValues(string(sys), string(Refused)).Inc()
	case dns.RcodeNameError:
		errorCount.WithLabelValues(string(sys), string(Nxdomain)).Inc()
		// nodata ??
	}

}

func ReportCacheMiss(ca CacheType) {
	if cacheMiss == nil {
		return
	}
	cacheMiss.WithLabelValues(string(ca)).Inc()
}

func envOrDefault(env, def string) string {
	e := os.Getenv(env)
	if e != "" {
		return e
	}
	return def
}
