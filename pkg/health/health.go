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
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

// Status represents the result of a single health-check operation.
type Status int

// Status values must be one of these constants.
const (
	Healthy Status = iota
	Unhealthy
	Unknown
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
