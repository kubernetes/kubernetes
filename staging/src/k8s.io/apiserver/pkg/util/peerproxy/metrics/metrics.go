/*
Copyright 2023 The Kubernetes Authors.

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
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	subsystem  = "apiserver"
	statuscode = "code"
)

var registerMetricsOnce sync.Once

var (
	// peerProxiedRequestsTotal counts the number of requests that were proxied to a peer kube-apiserver.
	peerProxiedRequestsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "rerouted_request_total",
			Help:           "Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{statuscode},
	)
)

func Register() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(peerProxiedRequestsTotal)
	})
}

// Only used for tests.
func Reset() {
	legacyregistry.Reset()
}

// IncPeerProxiedRequest increments the # of proxied requests to peer kube-apiserver
func IncPeerProxiedRequest(ctx context.Context, status string) {
	peerProxiedRequestsTotal.WithContext(ctx).WithLabelValues(status).Add(1)
}
