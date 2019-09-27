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

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var proxierHealthzRetryInterval = 60 * time.Second

// ProxierHealthUpdater allows callers to update healthz timestamp only.
type ProxierHealthUpdater interface {
	UpdateTimestamp()
}

// ProxierHealthServer returns 200 "OK" by default. Once timestamp has been
// updated, it verifies we don't exceed max no respond duration since
// last update.
type ProxierHealthServer struct {
	listener    listener
	httpFactory httpServerFactory
	clock       clock.Clock

	addr          string
	port          int32
	healthTimeout time.Duration
	recorder      record.EventRecorder
	nodeRef       *v1.ObjectReference

	lastUpdated atomic.Value
}

// NewProxierHealthServer returns a proxier health http server.
func NewProxierHealthServer(addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) *ProxierHealthServer {
	return newProxierHealthServer(stdNetListener{}, stdHTTPServerFactory{}, clock.RealClock{}, addr, healthTimeout, recorder, nodeRef)
}

func newProxierHealthServer(listener listener, httpServerFactory httpServerFactory, c clock.Clock, addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) *ProxierHealthServer {
	return &ProxierHealthServer{
		listener:      listener,
		httpFactory:   httpServerFactory,
		clock:         c,
		addr:          addr,
		healthTimeout: healthTimeout,
		recorder:      recorder,
		nodeRef:       nodeRef,
	}
}

// UpdateTimestamp updates the lastUpdated timestamp.
func (hs *ProxierHealthServer) UpdateTimestamp() {
	hs.lastUpdated.Store(hs.clock.Now())
}

// Run starts the healthz http server and returns.
func (hs *ProxierHealthServer) Run() {
	serveMux := http.NewServeMux()
	serveMux.Handle("/healthz", healthzHandler{hs: hs})
	server := hs.httpFactory.New(hs.addr, serveMux)

	go wait.Until(func() {
		klog.V(3).Infof("Starting goroutine for proxier healthz on %s", hs.addr)

		listener, err := hs.listener.Listen(hs.addr)
		if err != nil {
			msg := fmt.Sprintf("Failed to start proxier healthz on %s: %v", hs.addr, err)
			if hs.recorder != nil {
				hs.recorder.Eventf(hs.nodeRef, api.EventTypeWarning, "FailedToStartProxierHealthcheck", msg)
			}
			klog.Error(msg)
			return
		}

		if err := server.Serve(listener); err != nil {
			klog.Errorf("Proxier healthz closed with error: %v", err)
			return
		}
		klog.Error("Unexpected proxier healthz closed.")
	}, proxierHealthzRetryInterval, wait.NeverStop)
}

type healthzHandler struct {
	hs *ProxierHealthServer
}

func (h healthzHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	lastUpdated := time.Time{}
	if val := h.hs.lastUpdated.Load(); val != nil {
		lastUpdated = val.(time.Time)
	}
	currentTime := h.hs.clock.Now()

	resp.Header().Set("Content-Type", "application/json")
	resp.Header().Set("X-Content-Type-Options", "nosniff")
	if !lastUpdated.IsZero() && currentTime.After(lastUpdated.Add(h.hs.healthTimeout)) {
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
		resp.WriteHeader(http.StatusOK)
	}
	fmt.Fprintf(resp, fmt.Sprintf(`{"lastUpdated": %q,"currentTime": %q}`, lastUpdated, currentTime))
}
