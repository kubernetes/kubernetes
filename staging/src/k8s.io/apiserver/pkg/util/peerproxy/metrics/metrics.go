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
	group      = "group"
	version    = "version"
	resource   = "resource"
	errorType  = "type"

	// ProxyErrorEndpointResolution indicates a failure to resolve the network address of a peer apiserver.
	ProxyErrorEndpointResolution = "endpoint_resolution"
	// ProxyErrorTransport indicates a failure to build the proxy transport for the request.
	ProxyErrorTransport = "proxy_transport"

	// DiscoveryErrorLeaseList indicates a failure to list apiserver identity leases.
	DiscoveryErrorLeaseList = "lease_list"
	// DiscoveryErrorHostPortResolution indicates a failure to resolve host/port from an identity lease.
	DiscoveryErrorHostPortResolution = "hostport_resolution"
	// DiscoveryErrorFetch indicates a failure to fetch discovery document from a peer.
	DiscoveryErrorFetch = "fetch_discovery"
)

var registerMetricsOnce sync.Once

var (
	// peerProxiedRequestsTotal counts the number of requests that were proxied to a peer kube-apiserver.
	peerProxiedRequestsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "rerouted_request_total",
			Help:           `Total number of requests that were proxied to a peer kube-apiserver because the local apiserver was not capable of serving it, broken down by 'group', 'version', and 'resource' indicating the GVR of the request. If all three are empty (""), the request is a discovery request.`,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{statuscode, group, version, resource},
	)

	// peerProxyErrorsTotal counts the number of errors encountered while proxying requests to a peer kube-apiserver.
	peerProxyErrorsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "peer_proxy_errors_total",
			Help:           "Total number of errors encountered while proxying requests to a peer kube apiserver",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{errorType, group, version, resource},
	)

	// peerDiscoverySyncErrorsTotal counts the number of errors encountered while syncing discovery information from a peer kube-apiserver.
	peerDiscoverySyncErrorsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "peer_discovery_sync_errors_total",
			Help:           "Total number of errors encountered while syncing discovery information from a peer kube-apiserver",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{errorType},
	)
)

func Register() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(peerProxiedRequestsTotal)
		legacyregistry.MustRegister(peerProxyErrorsTotal)
		legacyregistry.MustRegister(peerDiscoverySyncErrorsTotal)
	})
}

// Only used for tests.
func Reset() {
	legacyregistry.Reset()
}

// IncPeerProxiedRequest increments the # of proxied requests to peer kube-apiserver
func IncPeerProxiedRequest(ctx context.Context, status, g, v, r string) {
	peerProxiedRequestsTotal.WithContext(ctx).WithLabelValues(status, g, v, r).Add(1)
}

// IncPeerProxyError increments the # of errors encountered during peer proxying
func IncPeerProxyError(ctx context.Context, e, g, v, r string) {
	peerProxyErrorsTotal.WithContext(ctx).WithLabelValues(e, g, v, r).Add(1)
}

// IncPeerDiscoverySyncError increments the # of errors encountered during peer discovery sync
func IncPeerDiscoverySyncError(ctx context.Context, e string) {
	peerDiscoverySyncErrorsTotal.WithContext(ctx).WithLabelValues(e).Add(1)
}
