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

package kubelet

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

type HealthCheckStatus int

const (
	CheckHealthy   HealthCheckStatus = 0
	CheckUnhealthy HealthCheckStatus = 1
	CheckUnknown   HealthCheckStatus = 2
)

type HealthChecker interface {
	HealthCheck(container api.Container) (HealthCheckStatus, error)
}

type httpDoInterface interface {
	Get(string) (*http.Response, error)
}

// MakeHealthChecker creates a new HealthChecker.
func MakeHealthChecker() HealthChecker {
	return &MuxHealthChecker{
		checkers: map[string]HealthChecker{
			"http": &HTTPHealthChecker{
				client: &http.Client{},
			},
		},
	}
}

// MuxHealthChecker bundles multiple implementations of HealthChecker of different types.
type MuxHealthChecker struct {
	checkers map[string]HealthChecker
}

func (m *MuxHealthChecker) HealthCheck(container api.Container) (HealthCheckStatus, error) {
	checker, ok := m.checkers[container.LivenessProbe.Type]
	if !ok || checker == nil {
		glog.Warningf("Failed to find health checker for %s %s", container.Name, container.LivenessProbe.Type)
		return CheckUnknown, nil
	}
	return checker.HealthCheck(container)
}

// HTTPHealthChecker is an implementation of HealthChecker which checks container health by sending HTTP Get requests.
type HTTPHealthChecker struct {
	client httpDoInterface
}

func (h *HTTPHealthChecker) findPort(container api.Container, portName string) int64 {
	for _, port := range container.Ports {
		if port.Name == portName {
			// TODO This means you can only health check exposed ports
			return int64(port.HostPort)
		}
	}
	return -1
}

func (h *HTTPHealthChecker) HealthCheck(container api.Container) (HealthCheckStatus, error) {
	params := container.LivenessProbe.HTTPGet
	if params == nil {
		return CheckUnknown, fmt.Errorf("Error, no HTTP parameters specified: %v", container)
	}
	port := h.findPort(container, params.Port)
	if port == -1 {
		var err error
		port, err = strconv.ParseInt(params.Port, 10, 0)
		if err != nil {
			return CheckUnknown, err
		}
	}
	var host string
	if len(params.Host) > 0 {
		host = params.Host
	} else {
		host = "localhost"
	}
	url := fmt.Sprintf("http://%s:%d%s", host, port, params.Path)
	res, err := h.client.Get(url)
	if res != nil && res.Body != nil {
		defer res.Body.Close()
	}
	if err != nil {
		// At this point, if it fails, its either a policy (unlikely) or HTTP protocol (likely) error.
		return CheckUnhealthy, nil
	}
	if res.StatusCode == http.StatusOK {
		return CheckHealthy, nil
	}
	glog.V(1).Infof("Health check failed for %v, Response: %v", container, *res)
	return CheckUnhealthy, nil
}
