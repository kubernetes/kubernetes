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
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	"k8s.io/utils/clock"
)

const (
	// ToBeDeletedTaint is a taint used by the CLuster Autoscaler before marking a node for deletion. Defined in
	// https://github.com/kubernetes/autoscaler/blob/e80ab518340f88f364fe3ef063f8303755125971/cluster-autoscaler/utils/deletetaint/delete.go#L36
	ToBeDeletedTaint = "ToBeDeletedByClusterAutoscaler"
)

// ProxierHealthUpdater allows callers to update healthz timestamp only.
type ProxierHealthUpdater interface {
	// QueuedUpdate should be called when the proxier receives a Service or Endpoints
	// event containing information that requires updating service rules.
	QueuedUpdate()

	// Updated should be called when the proxier has successfully updated the service
	// rules to reflect the current state.
	Updated()

	// Run starts the healthz HTTP server and blocks until it exits.
	Run() error

	// Sync the node and determine if its eligible or not. Eligible is
	// defined as being: not tainted by ToBeDeletedTaint and not deleted.
	SyncNode(node *v1.Node)

	proxierHealthChecker
}

var _ ProxierHealthUpdater = &proxierHealthServer{}
var zeroTime = time.Time{}

// proxierHealthServer returns 200 "OK" by default. It verifies that the delay between
// QueuedUpdate() calls and Updated() calls never exceeds healthTimeout.
type proxierHealthServer struct {
	listener    listener
	httpFactory httpServerFactory
	clock       clock.Clock

	addr          string
	healthTimeout time.Duration
	recorder      events.EventRecorder
	nodeRef       *v1.ObjectReference

	lastUpdated         atomic.Value
	oldestPendingQueued atomic.Value
	nodeEligible        atomic.Bool
}

// NewProxierHealthServer returns a proxier health http server.
func NewProxierHealthServer(addr string, healthTimeout time.Duration, recorder events.EventRecorder, nodeRef *v1.ObjectReference) ProxierHealthUpdater {
	return newProxierHealthServer(stdNetListener{}, stdHTTPServerFactory{}, clock.RealClock{}, addr, healthTimeout, recorder, nodeRef)
}

func newProxierHealthServer(listener listener, httpServerFactory httpServerFactory, c clock.Clock, addr string, healthTimeout time.Duration, recorder events.EventRecorder, nodeRef *v1.ObjectReference) *proxierHealthServer {
	hs := &proxierHealthServer{
		listener:      listener,
		httpFactory:   httpServerFactory,
		clock:         c,
		addr:          addr,
		healthTimeout: healthTimeout,
		recorder:      recorder,
		nodeRef:       nodeRef,
	}
	// The node is eligible (and thus the proxy healthy) while it's starting up
	// and until we've processed the first node event that indicates the
	// contrary.
	hs.nodeEligible.Store(true)
	return hs
}

// Updated indicates that kube-proxy has successfully updated its backend, so it should
// be considered healthy now.
func (hs *proxierHealthServer) Updated() {
	hs.oldestPendingQueued.Store(zeroTime)
	hs.lastUpdated.Store(hs.clock.Now())
}

// QueuedUpdate indicates that the proxy has received changes from the apiserver but
// has not yet pushed them to its backend. If the proxy does not call Updated within the
// healthTimeout time then it will be considered unhealthy.
func (hs *proxierHealthServer) QueuedUpdate() {
	// Set oldestPendingQueued only if it's currently zero
	hs.oldestPendingQueued.CompareAndSwap(zeroTime, hs.clock.Now())
}

// IsHealthy returns only the proxier's health state, following the same
// definition the HTTP server defines, but ignoring the state of the Node.
func (hs *proxierHealthServer) IsHealthy() bool {
	isHealthy, _, _ := hs.isHealthy()
	return isHealthy
}

func (hs *proxierHealthServer) isHealthy() (bool, time.Time, time.Time) {
	var oldestPendingQueued, lastUpdated time.Time
	if val := hs.oldestPendingQueued.Load(); val != nil {
		oldestPendingQueued = val.(time.Time)
	}
	if val := hs.lastUpdated.Load(); val != nil {
		lastUpdated = val.(time.Time)
	}
	currentTime := hs.clock.Now()

	healthy := false
	switch {
	case oldestPendingQueued.IsZero():
		// The proxy is healthy while it's starting up
		// or the proxy is fully synced.
		healthy = true
	case currentTime.Sub(oldestPendingQueued) < hs.healthTimeout:
		// There's an unprocessed update queued, but it's not late yet
		healthy = true
	}
	return healthy, lastUpdated, currentTime
}

func (hs *proxierHealthServer) SyncNode(node *v1.Node) {
	if !node.DeletionTimestamp.IsZero() {
		hs.nodeEligible.Store(false)
		return
	}
	for _, taint := range node.Spec.Taints {
		if taint.Key == ToBeDeletedTaint {
			hs.nodeEligible.Store(false)
			return
		}
	}
	hs.nodeEligible.Store(true)
}

// Run starts the healthz HTTP server and blocks until it exits.
func (hs *proxierHealthServer) Run() error {
	serveMux := http.NewServeMux()
	serveMux.Handle("/healthz", healthzHandler{hs: hs})
	serveMux.Handle("/livez", livezHandler{hs: hs})
	server := hs.httpFactory.New(hs.addr, serveMux)

	listener, err := hs.listener.Listen(hs.addr)
	if err != nil {
		msg := fmt.Sprintf("failed to start proxier healthz on %s: %v", hs.addr, err)
		// TODO(thockin): move eventing back to caller
		if hs.recorder != nil {
			hs.recorder.Eventf(hs.nodeRef, nil, api.EventTypeWarning, "FailedToStartProxierHealthcheck", "StartKubeProxy", msg)
		}
		return fmt.Errorf("%v", msg)
	}

	klog.V(3).InfoS("Starting healthz HTTP server", "address", hs.addr)

	if err := server.Serve(listener); err != nil {
		return fmt.Errorf("proxier healthz closed with error: %v", err)
	}
	return nil
}

type healthzHandler struct {
	hs *proxierHealthServer
}

func (h healthzHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	nodeEligible := h.hs.nodeEligible.Load()
	healthy, lastUpdated, currentTime := h.hs.isHealthy()
	healthy = healthy && nodeEligible
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
		lastUpdated = currentTime
	}
	fmt.Fprintf(resp, `{"lastUpdated": %q,"currentTime": %q, "nodeEligible": %v}`, lastUpdated, currentTime, nodeEligible)
}

type livezHandler struct {
	hs *proxierHealthServer
}

func (h livezHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	healthy, lastUpdated, currentTime := h.hs.isHealthy()
	resp.Header().Set("Content-Type", "application/json")
	resp.Header().Set("X-Content-Type-Options", "nosniff")
	if !healthy {
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
		lastUpdated = currentTime
	}
	fmt.Fprintf(resp, `{"lastUpdated": %q,"currentTime": %q}`, lastUpdated, currentTime)
}
