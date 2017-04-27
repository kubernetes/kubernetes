// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

package grpc_prometheus

import (
	"time"

	"google.golang.org/grpc/codes"

	prom "github.com/prometheus/client_golang/prometheus"
)

var (
	clientStartedCounter = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "client",
			Name:      "started_total",
			Help:      "Total number of RPCs started on the client.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	clientHandledCounter = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "client",
			Name:      "handled_total",
			Help:      "Total number of RPCs completed by the client, regardless of success or failure.",
		}, []string{"grpc_type", "grpc_service", "grpc_method", "grpc_code"})

	clientStreamMsgReceived = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "client",
			Name:      "msg_received_total",
			Help:      "Total number of RPC stream messages received by the client.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	clientStreamMsgSent = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "client",
			Name:      "msg_sent_total",
			Help:      "Total number of gRPC stream messages sent by the client.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	clientHandledHistogramEnabled = false
	clientHandledHistogramOpts    = prom.HistogramOpts{
		Namespace: "grpc",
		Subsystem: "client",
		Name:      "handling_seconds",
		Help:      "Histogram of response latency (seconds) of the gRPC until it is finished by the application.",
		Buckets:   prom.DefBuckets,
	}
	clientHandledHistogram *prom.HistogramVec
)

func init() {
	prom.MustRegister(clientStartedCounter)
	prom.MustRegister(clientHandledCounter)
	prom.MustRegister(clientStreamMsgReceived)
	prom.MustRegister(clientStreamMsgSent)
}

// EnableClientHandlingTimeHistogram turns on recording of handling time of RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func EnableClientHandlingTimeHistogram(opts ...HistogramOption) {
	for _, o := range opts {
		o(&clientHandledHistogramOpts)
	}
	if !clientHandledHistogramEnabled {
		clientHandledHistogram = prom.NewHistogramVec(
			clientHandledHistogramOpts,
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
		prom.Register(clientHandledHistogram)
	}
	clientHandledHistogramEnabled = true
}

type clientReporter struct {
	rpcType     grpcType
	serviceName string
	methodName  string
	startTime   time.Time
}

func newClientReporter(rpcType grpcType, fullMethod string) *clientReporter {
	r := &clientReporter{rpcType: rpcType}
	if clientHandledHistogramEnabled {
		r.startTime = time.Now()
	}
	r.serviceName, r.methodName = splitMethodName(fullMethod)
	clientStartedCounter.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
	return r
}

func (r *clientReporter) ReceivedMessage() {
	clientStreamMsgReceived.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
}

func (r *clientReporter) SentMessage() {
	clientStreamMsgSent.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
}

func (r *clientReporter) Handled(code codes.Code) {
	clientHandledCounter.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName, code.String()).Inc()
	if clientHandledHistogramEnabled {
		clientHandledHistogram.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Observe(time.Since(r.startTime).Seconds())
	}
}
