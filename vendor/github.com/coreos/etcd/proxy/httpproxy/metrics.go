// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package httpproxy

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	requestsIncoming = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "proxy",
			Name:      "requests_total",
			Help:      "Counter requests incoming by method.",
		}, []string{"method"})

	requestsHandled = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "proxy",
			Name:      "handled_total",
			Help:      "Counter of requests fully handled (by authoratitave servers)",
		}, []string{"method", "code"})

	requestsDropped = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "proxy",
			Name:      "dropped_total",
			Help:      "Counter of requests dropped on the proxy.",
		}, []string{"method", "proxying_error"})

	requestsHandlingTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "proxy",
			Name:      "handling_duration_seconds",
			Help: "Bucketed histogram of handling time of successful events (non-watches), by method " +
				"(GET/PUT etc.).",
			Buckets: prometheus.ExponentialBuckets(0.0005, 2, 13),
		}, []string{"method"})
)

type forwardingError string

const (
	zeroEndpoints         forwardingError = "zero_endpoints"
	failedSendingRequest  forwardingError = "failed_sending_request"
	failedGettingResponse forwardingError = "failed_getting_response"
)

func init() {
	prometheus.MustRegister(requestsIncoming)
	prometheus.MustRegister(requestsHandled)
	prometheus.MustRegister(requestsDropped)
	prometheus.MustRegister(requestsHandlingTime)
}

func reportIncomingRequest(request *http.Request) {
	requestsIncoming.WithLabelValues(request.Method).Inc()
}

func reportRequestHandled(request *http.Request, response *http.Response, startTime time.Time) {
	method := request.Method
	requestsHandled.WithLabelValues(method, strconv.Itoa(response.StatusCode)).Inc()
	requestsHandlingTime.WithLabelValues(method).Observe(time.Since(startTime).Seconds())
}

func reportRequestDropped(request *http.Request, err forwardingError) {
	requestsDropped.WithLabelValues(request.Method, string(err)).Inc()
}
