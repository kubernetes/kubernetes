// Copyright 2015 CoreOS, Inc.
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

package v2http

import (
	"strconv"
	"time"

	"net/http"

	etcdErr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	incomingEvents = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "http",
			Name:      "received_total",
			Help:      "Counter of requests received into the system (successfully parsed and authd).",
		}, []string{"method"})

	failedEvents = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "http",
			Name:      "failed_total",
			Help:      "Counter of handle failures of requests (non-watches), by method (GET/PUT etc.) and code (400, 500 etc.).",
		}, []string{"method", "code"})

	successfulEventsHandlingTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "http",
			Name:      "successful_duration_second",
			Help:      "Bucketed histogram of processing time (s) of successfully handled requests (non-watches), by method (GET/PUT etc.).",
			Buckets:   prometheus.ExponentialBuckets(0.0005, 2, 13),
		}, []string{"method"})
)

func init() {
	prometheus.MustRegister(incomingEvents)
	prometheus.MustRegister(failedEvents)
	prometheus.MustRegister(successfulEventsHandlingTime)
}

func reportRequestReceived(request etcdserverpb.Request) {
	incomingEvents.WithLabelValues(methodFromRequest(request)).Inc()
}

func reportRequestCompleted(request etcdserverpb.Request, response etcdserver.Response, startTime time.Time) {
	method := methodFromRequest(request)
	successfulEventsHandlingTime.WithLabelValues(method).Observe(time.Since(startTime).Seconds())
}

func reportRequestFailed(request etcdserverpb.Request, err error) {
	method := methodFromRequest(request)
	failedEvents.WithLabelValues(method, strconv.Itoa(codeFromError(err))).Inc()
}

func methodFromRequest(request etcdserverpb.Request) string {
	if request.Method == "GET" && request.Quorum {
		return "QGET"
	}
	return request.Method
}

func codeFromError(err error) int {
	if err == nil {
		return http.StatusInternalServerError
	}
	switch e := err.(type) {
	case *etcdErr.Error:
		return (*etcdErr.Error)(e).StatusCode()
	case *httptypes.HTTPError:
		return (*httptypes.HTTPError)(e).Code
	default:
		return http.StatusInternalServerError
	}
}
