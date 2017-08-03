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
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
)

var nodeHealthzRetryInterval = 60 * time.Second

// Server serves HTTP endpoints for each service name, with results
// based on the endpoints.  If there are 0 endpoints for a service, it returns a
// 503 "Service Unavailable" error (telling LBs not to use this node).  If there
// are 1 or more endpoints, it returns a 200 "OK".
type Server interface {
	// Make the new set of services be active.  Services that were open before
	// will be closed.  Services that are new will be opened.  Service that
	// existed and are in the new set will be left alone.  The value of the map
	// is the healthcheck-port to listen on.
	SyncServices(newServices map[types.NamespacedName]uint16) error
	// Make the new set of endpoints be active.  Endpoints for services that do
	// not exist will be dropped.  The value of the map is the number of
	// endpoints the service has on this node.
	SyncEndpoints(newEndpoints map[types.NamespacedName]int) error
}

// Listener allows for testing of Server.  If the Listener argument
// to NewServer() is nil, the real net.Listen function will be used.
type Listener interface {
	// Listen is very much like net.Listen, except the first arg (network) is
	// fixed to be "tcp".
	Listen(addr string) (net.Listener, error)
}

// HTTPServerFactory allows for testing of Server.  If the
// HTTPServerFactory argument to NewServer() is nil, the real
// http.Server type will be used.
type HTTPServerFactory interface {
	// New creates an instance of a type satisfying HTTPServer.  This is
	// designed to include http.Server.
	New(addr string, handler http.Handler) HTTPServer
}

// HTTPServer allows for testing of Server.
type HTTPServer interface {
	// Server is designed so that http.Server satifies this interface,
	Serve(listener net.Listener) error
}

// NewServer allocates a new healthcheck server manager.  If either
// of the injected arguments are nil, defaults will be used.
func NewServer(hostname string, recorder record.EventRecorder, listener Listener, httpServerFactory HTTPServerFactory) Server {
	if listener == nil {
		listener = stdNetListener{}
	}
	if httpServerFactory == nil {
		httpServerFactory = stdHTTPServerFactory{}
	}
	return &server{
		hostname:    hostname,
		recorder:    recorder,
		listener:    listener,
		httpFactory: httpServerFactory,
		services:    map[types.NamespacedName]*hcInstance{},
	}
}

// Implement Listener in terms of net.Listen.
type stdNetListener struct{}

func (stdNetListener) Listen(addr string) (net.Listener, error) {
	return net.Listen("tcp", addr)
}

var _ Listener = stdNetListener{}

// Implement HTTPServerFactory in terms of http.Server.
type stdHTTPServerFactory struct{}

func (stdHTTPServerFactory) New(addr string, handler http.Handler) HTTPServer {
	return &http.Server{
		Addr:    addr,
		Handler: handler,
	}
}

var _ HTTPServerFactory = stdHTTPServerFactory{}

type server struct {
	hostname    string
	recorder    record.EventRecorder // can be nil
	listener    Listener
	httpFactory HTTPServerFactory

	lock     sync.Mutex
	services map[types.NamespacedName]*hcInstance
}

func (hcs *server) SyncServices(newServices map[types.NamespacedName]uint16) error {
	hcs.lock.Lock()
	defer hcs.lock.Unlock()

	// Remove any that are not needed any more.
	for nsn, svc := range hcs.services {
		if port, found := newServices[nsn]; !found || port != svc.port {
			glog.V(2).Infof("Closing healthcheck %q on port %d", nsn.String(), svc.port)
			if err := svc.listener.Close(); err != nil {
				glog.Errorf("Close(%v): %v", svc.listener.Addr(), err)
			}
			delete(hcs.services, nsn)
		}
	}

	// Add any that are needed.
	for nsn, port := range newServices {
		if hcs.services[nsn] != nil {
			glog.V(3).Infof("Existing healthcheck %q on port %d", nsn.String(), port)
			continue
		}

		glog.V(2).Infof("Opening healthcheck %q on port %d", nsn.String(), port)
		svc := &hcInstance{port: port}
		addr := fmt.Sprintf(":%d", port)
		svc.server = hcs.httpFactory.New(addr, hcHandler{name: nsn, hcs: hcs})
		var err error
		svc.listener, err = hcs.listener.Listen(addr)
		if err != nil {
			msg := fmt.Sprintf("node %s failed to start healthcheck %q on port %d: %v", hcs.hostname, nsn.String(), port, err)

			if hcs.recorder != nil {
				hcs.recorder.Eventf(
					&v1.ObjectReference{
						Kind:      "Service",
						Namespace: nsn.Namespace,
						Name:      nsn.Name,
						UID:       types.UID(nsn.String()),
					}, api.EventTypeWarning, "FailedToStartServiceHealthcheck", msg)
			}
			glog.Error(msg)
			continue
		}
		hcs.services[nsn] = svc

		go func(nsn types.NamespacedName, svc *hcInstance) {
			// Serve() will exit when the listener is closed.
			glog.V(3).Infof("Starting goroutine for healthcheck %q on port %d", nsn.String(), svc.port)
			if err := svc.server.Serve(svc.listener); err != nil {
				glog.V(3).Infof("Healthcheck %q closed: %v", nsn.String(), err)
				return
			}
			glog.V(3).Infof("Healthcheck %q closed", nsn.String())
		}(nsn, svc)
	}
	return nil
}

type hcInstance struct {
	port      uint16
	listener  net.Listener
	server    HTTPServer
	endpoints int // number of local endpoints for a service
}

type hcHandler struct {
	name types.NamespacedName
	hcs  *server
}

var _ http.Handler = hcHandler{}

func (h hcHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	h.hcs.lock.Lock()
	svc, ok := h.hcs.services[h.name]
	if !ok || svc == nil {
		h.hcs.lock.Unlock()
		glog.Errorf("Received request for closed healthcheck %q", h.name.String())
		return
	}
	count := svc.endpoints
	h.hcs.lock.Unlock()

	resp.Header().Set("Content-Type", "application/json")
	if count == 0 {
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
		resp.WriteHeader(http.StatusOK)
	}
	fmt.Fprintf(resp, strings.Trim(dedent.Dedent(fmt.Sprintf(`
		{
			"service": {
				"namespace": %q,
				"name": %q
			},
			"localEndpoints": %d
		}
		`, h.name.Namespace, h.name.Name, count)), "\n"))
}

func (hcs *server) SyncEndpoints(newEndpoints map[types.NamespacedName]int) error {
	hcs.lock.Lock()
	defer hcs.lock.Unlock()

	for nsn, count := range newEndpoints {
		if hcs.services[nsn] == nil {
			glog.V(3).Infof("Not saving endpoints for unknown healthcheck %q", nsn.String())
			continue
		}
		glog.V(3).Infof("Reporting %d endpoints for healthcheck %q", count, nsn.String())
		hcs.services[nsn].endpoints = count
	}
	for nsn, hci := range hcs.services {
		if _, found := newEndpoints[nsn]; !found {
			hci.endpoints = 0
		}
	}
	return nil
}

// HealthzUpdater allows callers to update healthz timestamp only.
type HealthzUpdater interface {
	UpdateTimestamp()
}

// HealthzServer returns 200 "OK" by default. Once timestamp has been
// updated, it verifies we don't exceed max no respond duration since
// last update.
type HealthzServer struct {
	listener    Listener
	httpFactory HTTPServerFactory
	clock       clock.Clock

	addr          string
	port          int32
	healthTimeout time.Duration
	recorder      record.EventRecorder
	nodeRef       *v1.ObjectReference

	lastUpdated atomic.Value
}

// NewDefaultHealthzServer returns a default healthz http server.
func NewDefaultHealthzServer(addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) *HealthzServer {
	return newHealthzServer(nil, nil, nil, addr, healthTimeout, recorder, nodeRef)
}

func newHealthzServer(listener Listener, httpServerFactory HTTPServerFactory, c clock.Clock, addr string, healthTimeout time.Duration, recorder record.EventRecorder, nodeRef *v1.ObjectReference) *HealthzServer {
	if listener == nil {
		listener = stdNetListener{}
	}
	if httpServerFactory == nil {
		httpServerFactory = stdHTTPServerFactory{}
	}
	if c == nil {
		c = clock.RealClock{}
	}
	return &HealthzServer{
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
func (hs *HealthzServer) UpdateTimestamp() {
	hs.lastUpdated.Store(hs.clock.Now())
}

// Run starts the healthz http server and returns.
func (hs *HealthzServer) Run() {
	serveMux := http.NewServeMux()
	serveMux.Handle("/healthz", healthzHandler{hs: hs})
	server := hs.httpFactory.New(hs.addr, serveMux)

	go wait.Until(func() {
		glog.V(3).Infof("Starting goroutine for healthz on %s", hs.addr)

		listener, err := hs.listener.Listen(hs.addr)
		if err != nil {
			msg := fmt.Sprintf("Failed to start node healthz on %s: %v", hs.addr, err)
			if hs.recorder != nil {
				hs.recorder.Eventf(hs.nodeRef, api.EventTypeWarning, "FailedToStartNodeHealthcheck", msg)
			}
			glog.Error(msg)
			return
		}

		if err := server.Serve(listener); err != nil {
			glog.Errorf("Healthz closed with error: %v", err)
			return
		}
		glog.Errorf("Unexpected healthz closed.")
	}, nodeHealthzRetryInterval, wait.NeverStop)
}

type healthzHandler struct {
	hs *HealthzServer
}

func (h healthzHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	lastUpdated := time.Time{}
	if val := h.hs.lastUpdated.Load(); val != nil {
		lastUpdated = val.(time.Time)
	}
	currentTime := h.hs.clock.Now()

	resp.Header().Set("Content-Type", "application/json")
	if !lastUpdated.IsZero() && currentTime.After(lastUpdated.Add(h.hs.healthTimeout)) {
		resp.WriteHeader(http.StatusServiceUnavailable)
	} else {
		resp.WriteHeader(http.StatusOK)
	}
	fmt.Fprintf(resp, fmt.Sprintf(`{"lastUpdated": %q,"currentTime": %q}`, lastUpdated, currentTime))
}
