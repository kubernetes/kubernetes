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
	"sync"

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
	HealthCheck(podFullName, podUUID string, currentState api.PodState, container api.Container) (Status, error)
	CanCheck(probe *api.LivenessProbe) bool
}

// protects allCheckers
var checkerLock = sync.Mutex{}
var allCheckers = []HealthChecker{}

// AddHealthChecker adds a health checker to the list of known HealthChecker objects.
// Any subsequent call to NewHealthChecker will know about this HealthChecker.
func AddHealthChecker(checker HealthChecker) {
	checkerLock.Lock()
	defer checkerLock.Unlock()
	allCheckers = append(allCheckers, checker)
}

// NewHealthChecker creates a new HealthChecker which supports multiple types of liveness probes.
func NewHealthChecker() HealthChecker {
	checkerLock.Lock()
	defer checkerLock.Unlock()
	return &muxHealthChecker{
		checkers: append([]HealthChecker{}, allCheckers...),
	}
}

// muxHealthChecker bundles multiple implementations of HealthChecker of different types.
type muxHealthChecker struct {
	// Given a LivenessProbe, cycle through each known checker and see if it supports
	// the specific kind of probe (by returning non-nil).
	checkers []HealthChecker
}

func (m *muxHealthChecker) findCheckerFor(probe *api.LivenessProbe) HealthChecker {
	for i := range m.checkers {
		if m.checkers[i].CanCheck(probe) {
			return m.checkers[i]
		}
	}
	return nil
}

// HealthCheck delegates the health-checking of the container to one of the bundled implementations.
// If there is no health checker that can check container it returns Unknown, nil.
func (m *muxHealthChecker) HealthCheck(podFullName, podUUID string, currentState api.PodState, container api.Container) (Status, error) {
	checker := m.findCheckerFor(container.LivenessProbe)
	if checker == nil {
		glog.Warningf("Failed to find health checker for %s %+v", container.Name, container.LivenessProbe)
		return Unknown, nil
	}
	return checker.HealthCheck(podFullName, podUUID, currentState, container)
}

func (m *muxHealthChecker) CanCheck(probe *api.LivenessProbe) bool {
	return m.findCheckerFor(probe) != nil
}

// findPortByName is a helper function to look up a port in a container by name.
// Returns the HostPort if found, -1 if not found.
func findPortByName(container api.Container, portName string) int {
	for _, port := range container.Ports {
		if port.Name == portName {
			return port.HostPort
		}
	}
	return -1
}

func (s Status) String() string {
	switch s {
	case Healthy:
		return "healthy"
	case Unhealthy:
		return "unhealthy"
	default:
		return "unknown"
	}
}
