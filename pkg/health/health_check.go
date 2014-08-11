/*
Copyright 2014 Google Inc. All rights reserved.

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

package health

import (
	"fmt"
	"net"
	"net/http"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// HealthChecker defines an abstract interface for checking container health.
type HealthChecker interface {
	HealthCheck(currentState api.PodState, container api.Container) (Status, error)
}

// NewHealthChecker creates a new HealthChecker which supports multiple types of liveness probes.
func NewHealthChecker() HealthChecker {
	return &MuxHealthChecker{
		checkers: map[string]HealthChecker{
			"http": &HTTPHealthChecker{
				client: &http.Client{},
			},
			"tcp": &TCPHealthChecker{},
		},
	}
}

// MuxHealthChecker bundles multiple implementations of HealthChecker of different types.
type MuxHealthChecker struct {
	checkers map[string]HealthChecker
}

// HealthCheck delegates the health-checking of the container to one of the bundled implementations.
// It chooses an implementation according to container.LivenessProbe.Type.
// If there is no matching health checker it returns Unknown, nil.
func (m *MuxHealthChecker) HealthCheck(currentState api.PodState, container api.Container) (Status, error) {
	checker, ok := m.checkers[container.LivenessProbe.Type]
	if !ok || checker == nil {
		glog.Warningf("Failed to find health checker for %s %s", container.Name, container.LivenessProbe.Type)
		return Unknown, nil
	}
	return checker.HealthCheck(currentState, container)
}

// HTTPGetInterface is an abstract interface for testability. It abstracts the interface of http.Client.Get.
type HTTPGetInterface interface {
	Get(url string) (*http.Response, error)
}

// DoHTTPCheck checks if a GET request to the url succeeds.
// HTTPHealthChecker is an implementation of HealthChecker which checks container health by sending HTTP Get requests.
type HTTPHealthChecker struct {
	client HTTPGetInterface
}

// A helper function to look up a port in a container by name.
// Returns the HostPort if found, -1 if not found.
func findPortByName(container api.Container, portName string) int {
	for _, port := range container.Ports {
		if port.Name == portName {
			return port.HostPort
		}
	}
	return -1
}

// Get the components of the target URL.  For testability.
func getURLParts(currentState api.PodState, container api.Container) (string, int, string, error) {
	params := container.LivenessProbe.HTTPGet
	if params == nil {
		return "", -1, "", fmt.Errorf("no HTTP parameters specified: %v", container)
	}
	port := -1
	switch params.Port.Kind {
	case util.IntstrInt:
		port = params.Port.IntVal
	case util.IntstrString:
		port = findPortByName(container, params.Port.StrVal)
		if port == -1 {
			// Last ditch effort - maybe it was an int stored as string?
			var err error
			if port, err = strconv.Atoi(params.Port.StrVal); err != nil {
				return "", -1, "", err
			}
		}
	}
	if port == -1 {
		return "", -1, "", fmt.Errorf("unknown port: %v", params.Port)
	}
	var host string
	if len(params.Host) > 0 {
		host = params.Host
	} else {
		host = currentState.PodIP
	}

	return host, port, params.Path, nil
}

// Formats a URL from args.  For testability.
func formatURL(host string, port int, path string) string {
	return fmt.Sprintf("http://%s:%d%s", host, port, path)
}

// If the HTTP response code is successful (i.e. 400 > code >= 200), it returns Healthy.
// If the HTTP response code is unsuccessful, it returns Unhealthy.
// It returns Unknown and err if the HTTP communication itself fails.
func DoHTTPCheck(url string, client HTTPGetInterface) (Status, error) {
	res, err := client.Get(url)
	if err != nil {
		return Unknown, err
	}
	defer res.Body.Close()
	if res.StatusCode >= http.StatusOK && res.StatusCode < http.StatusBadRequest {
		return Healthy, nil
	}
	glog.V(1).Infof("Health check failed for %s, Response: %v", url, *res)
	return Unhealthy, nil
}

// HealthCheck checks if the container is healthy by trying sending HTTP Get requests to the container.
func (h *HTTPHealthChecker) HealthCheck(currentState api.PodState, container api.Container) (Status, error) {
	host, port, path, err := getURLParts(currentState, container)
	if err != nil {
		return Unknown, err
	}
	return DoHTTPCheck(formatURL(host, port, path), h.client)
}

type TCPHealthChecker struct{}

// Get the components of a TCP connection address.  For testability.
func getTCPAddrParts(currentState api.PodState, container api.Container) (string, int, error) {
	params := container.LivenessProbe.TCPSocket
	if params == nil {
		return "", -1, fmt.Errorf("error, no TCP parameters specified: %v", container)
	}
	port := -1
	switch params.Port.Kind {
	case util.IntstrInt:
		port = params.Port.IntVal
	case util.IntstrString:
		port = findPortByName(container, params.Port.StrVal)
		if port == -1 {
			// Last ditch effort - maybe it was an int stored as string?
			var err error
			if port, err = strconv.Atoi(params.Port.StrVal); err != nil {
				return "", -1, err
			}
		}
	}
	if port == -1 {
		return "", -1, fmt.Errorf("unknown port: %v", params.Port)
	}
	if len(currentState.PodIP) == 0 {
		return "", -1, fmt.Errorf("no host specified.")
	}

	return currentState.PodIP, port, nil
}

// DoTCPCheck checks that a TCP socket to the address can be opened.
// If the socket can be opened, it returns Healthy.
// If the socket fails to open, it returns Unhealthy.
func DoTCPCheck(addr string) (Status, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return Unhealthy, nil
	}
	err = conn.Close()
	if err != nil {
		glog.Errorf("unexpected error closing health check socket: %v (%#v)", err, err)
	}
	return Healthy, nil
}

func (t *TCPHealthChecker) HealthCheck(currentState api.PodState, container api.Container) (Status, error) {
	host, port, err := getTCPAddrParts(currentState, container)
	if err != nil {
		return Unknown, err
	}
	return DoTCPCheck(net.JoinHostPort(host, strconv.Itoa(port)))
}
