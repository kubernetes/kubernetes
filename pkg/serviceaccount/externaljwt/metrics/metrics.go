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

package metrics

import (
	"context"
	"errors"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "externaljwt"
)

var (
	lastKeyFetchTimeStamp = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "fetch_keys_success_timestamp",
			Help:           "Unix Timestamp in seconds of the last successful FetchKeys request",
			StabilityLevel: metrics.ALPHA,
		},
		nil,
	)

	dataTimeStamp = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "fetch_keys_data_timestamp",
			Help:           "Unix Timestamp in seconds of the last successful FetchKeys data_timestamp value returned by the external signer",
			StabilityLevel: metrics.ALPHA,
		},
		nil,
	)

	totalKeyFetch = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "fetch_keys_request_total",
			Help:           "Total attempts at syncing supported JWKs",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)

	tokenGenReqTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "sign_request_total",
			Help:           "Total attempts at signing JWT",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)

	requestDurationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "request_duration_seconds",
			Help:           "Request duration and time for calls to external-jwt-signer",
			Buckets:        []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"method", "code"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(lastKeyFetchTimeStamp)
		legacyregistry.MustRegister(dataTimeStamp)
		legacyregistry.MustRegister(totalKeyFetch)
		legacyregistry.MustRegister(tokenGenReqTotal)
		legacyregistry.MustRegister(requestDurationSeconds)
	})
}

func RecordFetchKeysAttempt(err error) {
	totalKeyFetch.WithLabelValues(getErrorCode(err)).Inc()
	if err == nil {
		lastKeyFetchTimeStamp.WithLabelValues().SetToCurrentTime()
	}
}

func RecordTokenGenAttempt(err error) {
	tokenGenReqTotal.WithLabelValues(getErrorCode(err)).Inc()
}

func RecordKeyDataTimeStamp(timestamp float64) {
	dataTimeStamp.WithLabelValues().Set(timestamp)
}

type gRPCError interface {
	GRPCStatus() *status.Status
}

func getErrorCode(err error) string {
	if err == nil {
		return codes.OK.String()
	}

	// handle errors wrapped with fmt.Errorf and similar
	var s gRPCError
	if errors.As(err, &s) {
		return s.GRPCStatus().Code().String()
	}

	// This is not gRPC error. The operation must have failed before gRPC
	// method was called, otherwise we would get gRPC error.
	return "unknown-non-grpc"
}

func OuboundRequestMetricsInterceptor(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	start := time.Now()
	err := invoker(ctx, method, req, reply, cc, opts...)
	requestDurationSeconds.WithLabelValues(method, getErrorCode(err)).Observe(time.Since(start).Seconds())
	return err
}
