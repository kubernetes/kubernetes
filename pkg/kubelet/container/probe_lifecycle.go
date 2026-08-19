/*
Copyright The Kubernetes Authors.

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

package container

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
)

// ProbeType identifies a probe kind (liveness, readiness, startup).
type ProbeType int

const (
	LivenessProbe ProbeType = 1 << iota
	ReadinessProbe
	StartupProbe

	AllProbes = LivenessProbe | ReadinessProbe | StartupProbe
)

func (t ProbeType) String() string {
	switch t {
	case ReadinessProbe:
		return "Readiness"
	case LivenessProbe:
		return "Liveness"
	case StartupProbe:
		return "Startup"
	case AllProbes:
		return "All"
	case LivenessProbe | StartupProbe:
		return "Liveness|Startup"
	default:
		return "UNKNOWN"
	}
}

// ContainerProbeLifecycle lets the container runtime tell the probe manager
// when the containers it probes start and stop, so that probing a container is
// tied to that container existing rather than being inferred after the fact.
//
// It lives here because the runtime manager (which calls it) and the prober
// (which implements it) both already depend on this package.
type ContainerProbeLifecycle interface {
	// StartProbes starts probing a container instance. It is called once CRI
	// StartContainer has succeeded, and must be idempotent for a given
	// container ID.
	StartProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID ContainerID, podIPs []string, startedAt time.Time)

	// StopProbes stops the specified probes for a container instance.
	// When AllProbes is specified, cached probe results for the container are also dropped.
	StopProbes(containerID ContainerID, probeTypes ProbeType)
}

// NoopContainerProbeLifecycle is a no-op implementation of ContainerProbeLifecycle.
type NoopContainerProbeLifecycle struct{}

var _ ContainerProbeLifecycle = NoopContainerProbeLifecycle{}

func (NoopContainerProbeLifecycle) StartProbes(context.Context, *v1.Pod, *v1.Container, ContainerID, []string, time.Time) {
}

func (NoopContainerProbeLifecycle) StopProbes(ContainerID, ProbeType) {}
