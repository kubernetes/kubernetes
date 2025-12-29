/*
Copyright 2016 The Kubernetes Authors.

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

package healthcheck

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const (
	// ToBeDeletedTaint is a taint used by the CLuster Autoscaler before marking a node for deletion. Defined in
	// https://github.com/kubernetes/autoscaler/blob/e80ab518340f88f364fe3ef063f8303755125971/cluster-autoscaler/utils/deletetaint/delete.go#L36
	ToBeDeletedTaint = "ToBeDeletedByClusterAutoscaler"
)

// ProxierHealth represents the health of a proxier which operates on a single IP family.
type ProxierHealth struct {
	LastUpdated time.Time `json:"lastUpdated"`
	Healthy     bool      `json:"healthy"`
}

// ProxyHealth represents the health of kube-proxy, embeds health of individual proxiers.
type ProxyHealth struct {
	// LastUpdated is the last updated time of the proxier
	// which was updated most recently.
	// This is kept for backward-compatibility.
	LastUpdated  time.Time `json:"lastUpdated"`
	CurrentTime  time.Time `json:"currentTime"`
	NodeEligible *bool     `json:"nodeEligible,omitempty"`
	// Healthy is true when all the proxiers are healthy,
	// false otherwise.
	Healthy bool `json:"healthy"`
	// status of the health check per IP family
	Status map[v1.IPFamily]ProxierHealth `json:"status,omitempty"`
}

// ProxyHealthServer allows callers to:
//  1. run a http server with /healthz and /livez endpoint handlers.
//  2. update healthz timestamps before and after synchronizing dataplane.
//  3. sync node status, for reporting unhealthy /healthz response
//     if the node is marked for deletion by autoscaler.
//  4. get proxy health by verifying that the delay between QueuedUpdate()
//     calls and Updated() calls exceeded healthTimeout or not.
type ProxyHealthServer struct {
	listener    listener
	httpFactory httpServerFactory
	clock       clock.Clock

	nodeManager nodeManager

	addr          string
	healthTimeout time.Duration

	lock                   sync.RWMutex
	lastUpdatedMap         map[v1.IPFamily]time.Time
	oldestPendingQueuedMap map[v1.IPFamily]time.Time
}

// interface to break an import chain between pkg/proxy and pkg/proxy/healthcheck
type nodeManager interface {
	Node() *v1.Node
}

// NewProxyHealthServer returns a proxy health http server.
func NewProxyHealthServer(addr string, healthTimeout time.Duration, nodeManager nodeManager) *ProxyHealthServer {
	return newProxyHealthServer(stdNetListener{}, stdHTTPServerFactory{}, clock.RealClock{}, addr, healthTimeout, nodeManager)
}

func newProxyHealthServer(listener listener, httpServerFactory httpServerFactory, c clock.Clock, addr string, healthTimeout time.Duration, nodeManager nodeManager) *ProxyHealthServer {
	return &ProxyHealthServer{
		listener:      listener,
		httpFactory:   httpServerFactory,
		clock:         c,
		addr:          addr,
		healthTimeout: healthTimeout,
		nodeManager:   nodeManager,

		lastUpdatedMap:         make(map[v1.IPFamily]time.Time),
		oldestPendingQueuedMap: make(map[v1.IPFamily]time.Time),
	}
}

// Updated should be called when the proxier of the given IP family has successfully updated
// the service rules to reflect the current state and should be considered healthy now.
func (hs *ProxyHealthServer) Updated(ipFamily v1.IPFamily) {
	hs.lock.Lock()
	defer hs.lock.Unlock()
	delete(hs.oldestPendingQueuedMap, ipFamily)
	hs.lastUpdatedMap[ipFamily] = hs.clock.Now()
}

// QueuedUpdate should be called when the proxier receives a Service or Endpoints event
// from API Server containing information that requires updating service rules. It
// indicates that the proxier for the given IP family has received changes but has not
// yet pushed them to its backend. If the proxier does not call Updated within the
// healthTimeout time then it will be considered unhealthy.
func (hs *ProxyHealthServer) QueuedUpdate(ipFamily v1.IPFamily) {
	hs.lock.Lock()
	defer hs.lock.Unlock()
	// Set oldestPendingQueuedMap[ipFamily] only if it's currently unset
	if _, set := hs.oldestPendingQueuedMap[ipFamily]; !set {
		hs.oldestPendingQueuedMap[ipFamily] = hs.clock.Now()
	}
}

// Health returns proxy health status.
func (hs *ProxyHealthServer) Health() ProxyHealth {
	var health = ProxyHealth{
		Healthy: true,
		Status:  make(map[v1.IPFamily]ProxierHealth),
	}
	hs.lock.RLock()
	defer hs.lock.RUnlock()

	var lastUpdated time.Time
	for ipFamily, proxierLastUpdated := range hs.lastUpdatedMap {
		if proxierLastUpdated.After(lastUpdated) {
			lastUpdated = proxierLastUpdated
		}
		// initialize the health status of each proxier
		// with healthy=true and the last updated time
		// of the proxier.
		health.Status[ipFamily] = ProxierHealth{
			LastUpdated: proxierLastUpdated,
			Healthy:     true,
		}
	}

	currentTime := hs.clock.Now()
	health.CurrentTime = currentTime
	for ipFamily, proxierLastUpdated := range hs.lastUpdatedMap {
		if _, set := hs.oldestPendingQueuedMap[ipFamily]; !set {
			// the proxier is healthy while it's starting up
			// or the proxier is fully synced.
			continue
		}

		if currentTime.Sub(hs.oldestPendingQueuedMap[ipFamily]) < hs.healthTimeout {
			// there's an unprocessed update queued for this proxier, but it's not late yet.
			continue
		}

		// mark the status unhealthy.
		health.Healthy = false
		health.Status[ipFamily] = ProxierHealth{
			LastUpdated: proxierLastUpdated,
			Healthy:     false,
		}
	}
	health.LastUpdated = lastUpdated
	return health
}

// NodeEligible returns if node is eligible or not. Eligible is defined
// as being: not tainted by ToBeDeletedTaint and not deleted.
func (hs *ProxyHealthServer) NodeEligible() bool {
	hs.lock.Lock()
	defer hs.lock.Unlock()

	node := hs.nodeManager.Node()
	if node == nil {
		return false
	}
	if !node.DeletionTimestamp.IsZero() {
		return false
	}
	for _, taint := range node.Spec.Taints {
		if taint.Key == ToBeDeletedTaint {
			return false
		}
	}
	return true
}

// Run starts the healthz HTTP server and blocks until it exits.
func (hs *ProxyHealthServer) Run(ctx context.Context) error {
	serveMux := http.NewServeMux()
	serveMux.Handle("/healthz", healthzHandler{hs: hs})
	serveMux.Handle("/livez", livezHandler{hs: hs})
	server := hs.httpFactory.New(serveMux)

	listener, err := hs.listener.Listen(ctx, hs.addr)
	if err != nil {
		return fmt.Errorf("failed to start proxy healthz on %s: %w", hs.addr, err)
	}

	klog.V(3).InfoS("Starting healthz HTTP server", "address", hs.addr)

	if err := server.Serve(listener); err != nil {
		return fmt.Errorf("proxy healthz closed with error: %w", err)
	}
	return nil
}

type healthzHandler struct {
	hs *ProxyHealthServer
}

func (h healthzHandler) ServeHTTP(resp http.ResponseWriter, _ *http.Request) {
	health := h.hs.Health()
	nodeEligible := h.hs.NodeEligible()
	healthy := health.Healthy && nodeEligible
	// updating the node eligibility here (outside of Health() call) as we only want responses
	// of /healthz calls (not /livez) to have that.
	health.NodeEligible = ptr.To(nodeEligible)

	resp.Header().Set("Content-Type", "application/json")
	resp.Header().Set("X-Content-Type-Options", "nosniff")
	if !healthy {
		metrics.ProxyHealthzTotal.WithLabelValues("503").Inc()
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
		metrics.ProxyHealthzTotal.WithLabelValues("200").Inc()
		resp.WriteHeader(http.StatusOK)
		// In older releases, the returned "lastUpdated" time indicated the last
		// time the proxier sync loop ran, even if nothing had changed. To
		// preserve compatibility, we use the same semantics: the returned
		// lastUpdated value is "recent" if the server is healthy. The kube-proxy
		// metrics provide more detailed information.
		health.LastUpdated = h.hs.clock.Now()
	}

	output, _ := json.Marshal(health)
	_, _ = fmt.Fprint(resp, string(output))
}

type livezHandler struct {
	hs *ProxyHealthServer
}

func (h livezHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	health := h.hs.Health()

	resp.Header().Set("Content-Type", "application/json")
	resp.Header().Set("X-Content-Type-Options", "nosniff")
	if !health.Healthy {
		metrics.ProxyLivezTotal.WithLabelValues("503").Inc()
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
		metrics.ProxyLivezTotal.WithLabelValues("200").Inc()
		resp.WriteHeader(http.StatusOK)
		// In older releases, the returned "lastUpdated" time indicated the last
		// time the proxier sync loop ran, even if nothing had changed. To
		// preserve compatibility, we use the same semantics: the returned
		// lastUpdated value is "recent" if the server is healthy. The kube-proxy
		// metrics provide more detailed information.
		health.LastUpdated = h.hs.clock.Now()
	}
	output, _ := json.Marshal(health)
	_, _ = fmt.Fprint(resp, string(output))
}
