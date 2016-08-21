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

// Create/Delete dynamic listeners on the required nodePorts

import (
	"fmt"
	"net"
	"net/http"

	"github.com/golang/glog"
)

// handleServiceListenerRequest: receive requests to add/remove service health check listening ports
func (h *proxyHC) handleServiceListenerRequest(req *proxyListenerRequest) bool {
	sr, serviceFound := h.serviceResponderMap[req.serviceName]
	if !req.add {
		if !serviceFound {
			return false
		}
		glog.Infof("Deleting HealthCheckListenPort for service %s port %d",
			req.serviceName, req.listenPort)
		delete(h.serviceResponderMap, req.serviceName)
		(*sr.listener).Close()
		return true
	} else if serviceFound {
		if req.listenPort == sr.listenPort {
			// Addition requested but responder for service already exists and port is unchanged
			return true
		}
		// Addition requested but responder for service already exists but the listen port has changed
		glog.Infof("HealthCheckListenPort for service %s changed from %d to %d - closing old listening port",
			req.serviceName, sr.listenPort, req.listenPort)
		delete(h.serviceResponderMap, req.serviceName)
		(*sr.listener).Close()
	}
	// Create a service responder object and start listening and serving on the provided port
	glog.V(2).Infof("Adding health check listener for service %s on nodePort %d", req.serviceName, req.listenPort)
	server := http.Server{
		Addr:    fmt.Sprintf(":%d", req.listenPort),
		Handler: healthCheckHandler{svcNsName: req.serviceName.String()},
	}
	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		glog.Warningf("FAILED to listen on address %s (%s)\n", server.Addr, err)
		return false
	}
	h.serviceResponderMap[req.serviceName] = serviceResponder{serviceName: req.serviceName,
		listenPort: req.listenPort,
		listener:   &listener,
		server:     &server}
	go func() {
		// Anonymous goroutine to block on Serve for this listen port - Serve will exit when the listener is closed
		glog.V(3).Infof("Goroutine blocking on serving health checks for %s on port %d", req.serviceName, req.listenPort)
		if err := server.Serve(listener); err != nil {
			glog.V(3).Infof("Proxy HealthCheck listen socket %d for service %s closed with error %s\n", req.listenPort, req.serviceName, err)
			return
		}
		glog.V(3).Infof("Proxy HealthCheck listen socket %d for service %s closed\n", req.listenPort, req.serviceName)
	}()
	return true
}
