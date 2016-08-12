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
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/healthcheckparser"
)

// HTTP Utility function to send the required statusCode and error text to a http.ResponseWriter object
func sendHealthCheckResponse(rw http.ResponseWriter, statusCode int, error string) {
	rw.Header().Set("Content-Type", "text/plain")
	rw.WriteHeader(statusCode)
	fmt.Fprint(rw, error)
}

// Utility function to parse an incoming http request, construct and send a health check message to the main loop
// Waits for responses from the main loop before responding to and closing the connection to the HTTP client.
func parseHttpRequest(req *http.Request) (string, error) {
	glog.V(3).Infof("Received Health Check on url %s", req.URL.String())
	// Sanity check and parse the healthcheck URL
	namespace, name, err := healthcheckparser.ParseURL(req.URL.String())
	if err != nil {
		glog.Info("Parse failure - cannot respond to malformed healthcheck URL")
		return "", err
	}
	glog.V(4).Infof("Parsed Healthcheck as service %s/%s", namespace, name)

	serviceName := types.NamespacedName{namespace, name}
	return serviceName.String(), nil
}

// ServeHTTP: Interface callback method for net.Listener Handlers
func (h *proxyHC) ServeHTTP(response http.ResponseWriter, req *http.Request) {
	// Grab the session guid from the URL and lookup in the fastmap
	serviceName, err := parseHttpRequest(req)
	if err != nil {
		sendHealthCheckResponse(response, http.StatusBadRequest, fmt.Sprintf("Parse error: %s", err))
	}
	glog.V(3).Infof("Received HC Request Service %s from Cloud Load Balancer", serviceName)
	healthchecker.handleHealthCheckRequest(response, serviceName)
}
