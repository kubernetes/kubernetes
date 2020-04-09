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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
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
}

var _ ProxierHealthUpdater = &proxierHealthServer{}

// proxierHealthServer returns 200 "OK" by default. It verifies that the delay between
// QueuedUpdate() calls and Updated() calls never exceeds healthTimeout.
type proxierHealthServer struct {
	listener    listener
	httpFactory httpServerFactory
	clock       clock.Clock

	addr          string
	healthTimeout time.Duration
	recorder      record.EventRecorder
	nodeRef       *v1.ObjectReference

	lastUpdated atomic.Value
	lastQueued  atomic.Value
}

// NewProxierHealthServer returns a proxier health http server.
func NewProxierHealthServer(addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) ProxierHealthUpdater {
	return newProxierHealthServer(stdNetListener{}, stdHTTPServerFactory{}, clock.RealClock{}, addr, healthTimeout, recorder, nodeRef)
}

func newProxierHealthServer(listener listener, httpServerFactory httpServerFactory, c clock.Clock, addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) *proxierHealthServer {
	return &proxierHealthServer{
		listener:      listener,
		httpFactory:   httpServerFactory,
		clock:         c,
		addr:          addr,
		healthTimeout: healthTimeout,
		recorder:      recorder,
		nodeRef:       nodeRef,
	}
}

// Updated updates the lastUpdated timestamp.
func (hs *proxierHealthServer) Updated() {
	hs.lastUpdated.Store(hs.clock.Now())
}

// QueuedUpdate updates the lastQueued timestamp.
func (hs *proxierHealthServer) QueuedUpdate() {
	hs.lastQueued.Store(hs.clock.Now())
}

// Run starts the healthz HTTP server and blocks until it exits.
func (hs *proxierHealthServer) Run() error {
	serveMux := http.NewServeMux()
	serveMux.Handle("/healthz", healthzHandler{hs: hs})
	server := hs.httpFactory.New(hs.addr, serveMux)

	listener, err := hs.listener.Listen(hs.addr)
	if err != nil {
		msg := fmt.Sprintf("failed to start proxier healthz on %s: %v", hs.addr, err)
		// TODO(thockin): move eventing back to caller
		if hs.recorder != nil {
			hs.recorder.Eventf(hs.nodeRef, api.EventTypeWarning, "FailedToStartProxierHealthcheck", msg)
		}
		return fmt.Errorf("%v", msg)
	}

	klog.V(3).Infof("starting healthz on %s", hs.addr)

	if err := server.Serve(listener); err != nil {
		return fmt.Errorf("proxier healthz closed with error: %v", err)
	}
	return nil
}

type healthzHandler struct {
	hs *proxierHealthServer
}

func (h healthzHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	var lastQueued, lastUpdated time.Time
	if val := h.hs.lastQueued.Load(); val != nil {
		lastQueued = val.(time.Time)
	}
	if val := h.hs.lastUpdated.Load(); val != nil {
		lastUpdated = val.(time.Time)
	}
	currentTime := h.hs.clock.Now()

	healthy := false
	switch {
	case lastUpdated.IsZero():
		// The proxy is healthy while it's starting up
		// TODO: this makes it useless as a readinessProbe. Consider changing
		// to only become healthy after the proxy is fully synced.
		healthy = true
	case lastUpdated.After(lastQueued):
		// We've processed all updates
		healthy = true
	case currentTime.Sub(lastQueued) < h.hs.healthTimeout:
		// There's an unprocessed update queued, but it's not late yet
		healthy = true
	}

	resp.Header().Set("Content-Type", "application/json")
	resp.Header().Set("X-Content-Type-Options", "nosniff")
	if !healthy {
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
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
