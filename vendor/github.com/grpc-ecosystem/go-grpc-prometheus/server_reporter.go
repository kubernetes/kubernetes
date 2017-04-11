// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

package grpc_prometheus

import (
	"time"

	"google.golang.org/grpc/codes"

	prom "github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
)

type grpcType string

const (
	Unary        grpcType = "unary"
	ClientStream grpcType = "client_stream"
	ServerStream grpcType = "server_stream"
	BidiStream   grpcType = "bidi_stream"
)

var (
	serverStartedCounter = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "server",
			Name:      "started_total",
			Help:      "Total number of RPCs started on the server.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	serverHandledCounter = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "server",
			Name:      "handled_total",
			Help:      "Total number of RPCs completed on the server, regardless of success or failure.",
		}, []string{"grpc_type", "grpc_service", "grpc_method", "grpc_code"})

	serverStreamMsgReceived = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "server",
			Name:      "msg_received_total",
			Help:      "Total number of RPC stream messages received on the server.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	serverStreamMsgSent = prom.NewCounterVec(
		prom.CounterOpts{
			Namespace: "grpc",
			Subsystem: "server",
			Name:      "msg_sent_total",
			Help:      "Total number of gRPC stream messages sent by the server.",
		}, []string{"grpc_type", "grpc_service", "grpc_method"})

	serverHandledHistogramEnabled = false
	serverHandledHistogramOpts    = prom.HistogramOpts{
		Namespace: "grpc",
		Subsystem: "server",
		Name:      "handling_seconds",
		Help:      "Histogram of response latency (seconds) of gRPC that had been application-level handled by the server.",
		Buckets:   prom.DefBuckets,
	}
	serverHandledHistogram *prom.HistogramVec
)

func init() {
	prom.MustRegister(serverStartedCounter)
	prom.MustRegister(serverHandledCounter)
	prom.MustRegister(serverStreamMsgReceived)
	prom.MustRegister(serverStreamMsgSent)
}

type HistogramOption func(*prom.HistogramOpts)

// WithHistogramBuckets allows you to specify custom bucket ranges for histograms if EnableHandlingTimeHistogram is on.
func WithHistogramBuckets(buckets []float64) HistogramOption {
	return func(o *prom.HistogramOpts) { o.Buckets = buckets }
}

// EnableHandlingTimeHistogram turns on recording of handling time of RPCs for server-side interceptors.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func EnableHandlingTimeHistogram(opts ...HistogramOption) {
	for _, o := range opts {
		o(&serverHandledHistogramOpts)
	}
	if !serverHandledHistogramEnabled {
		serverHandledHistogram = prom.NewHistogramVec(
			serverHandledHistogramOpts,
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
		prom.Register(serverHandledHistogram)
	}
	serverHandledHistogramEnabled = true
}

type serverReporter struct {
	rpcType     grpcType
	serviceName string
	methodName  string
	startTime   time.Time
}

func newServerReporter(rpcType grpcType, fullMethod string) *serverReporter {
	r := &serverReporter{rpcType: rpcType}
	if serverHandledHistogramEnabled {
		r.startTime = time.Now()
	}
	r.serviceName, r.methodName = splitMethodName(fullMethod)
	serverStartedCounter.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
	return r
}

func (r *serverReporter) ReceivedMessage() {
	serverStreamMsgReceived.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
}

func (r *serverReporter) SentMessage() {
	serverStreamMsgSent.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Inc()
}

func (r *serverReporter) Handled(code codes.Code) {
	serverHandledCounter.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName, code.String()).Inc()
	if serverHandledHistogramEnabled {
		serverHandledHistogram.WithLabelValues(string(r.rpcType), r.serviceName, r.methodName).Observe(time.Since(r.startTime).Seconds())
	}
}

// preRegisterMethod is invoked on Register of a Server, allowing all gRPC services labels to be pre-populated.
func preRegisterMethod(serviceName string, mInfo *grpc.MethodInfo) {
	methodName := mInfo.Name
	methodType := string(typeFromMethodInfo(mInfo))
	// These are just references (no increments), as just referencing will create the labels but not set values.
	serverStartedCounter.GetMetricWithLabelValues(methodType, serviceName, methodName)
	serverStreamMsgReceived.GetMetricWithLabelValues(methodType, serviceName, methodName)
	serverStreamMsgSent.GetMetricWithLabelValues(methodType, serviceName, methodName)
	if serverHandledHistogramEnabled {
		serverHandledHistogram.GetMetricWithLabelValues(methodType, serviceName, methodName)
	}
	for _, code := range allCodes {
		serverHandledCounter.GetMetricWithLabelValues(methodType, serviceName, methodName, code.String())
	}
}

func typeFromMethodInfo(mInfo *grpc.MethodInfo) grpcType {
	if mInfo.IsClientStream == false && mInfo.IsServerStream == false {
		return Unary
	}
	if mInfo.IsClientStream == true && mInfo.IsServerStream == false {
		return ClientStream
	}
	if mInfo.IsClientStream == false && mInfo.IsServerStream == true {
		return ServerStream
	}
	return BidiStream
}
