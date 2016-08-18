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

	"github.com/golang/glog"
)

// A healthCheckHandler serves http requests on /healthz on the service health check node port,
// and responds to every request with either:
// 200 OK and the count of endpoints for the given service that are local to this node.
// or
// 503 Service Unavailable If the count is zero or the service does not exist
type healthCheckHandler struct {
	svcNsName string
}

// HTTP Utility function to send the required statusCode and error text to a http.ResponseWriter object
func sendHealthCheckResponse(rw http.ResponseWriter, statusCode int, error string) {
	rw.Header().Set("Content-Type", "text/plain")
	rw.WriteHeader(statusCode)
	fmt.Fprint(rw, error)
}

// ServeHTTP: Interface callback method for net.Listener Handlers
func (h healthCheckHandler) ServeHTTP(response http.ResponseWriter, req *http.Request) {
	glog.V(4).Infof("Received HC Request Service %s from Cloud Load Balancer", h.svcNsName)
	healthchecker.handleHealthCheckRequest(response, h.svcNsName)
}
