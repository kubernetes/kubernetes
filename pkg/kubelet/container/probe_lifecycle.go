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

// ContainerProbeLifecycle notifies the probe manager of container start and stop
// events to bind probe lifecycles directly to container instances.
type ContainerProbeLifecycle interface {
	// StartProbes starts probing a container instance once CRI StartContainer
	// succeeds. Must be idempotent for a given container ID.
	StartProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID ContainerID, podIPs []string, startedAt time.Time)

	// StopProbes stops the specified probes for a container instance and drops
	// cached results when AllProbes is specified.
	StopProbes(containerID ContainerID, probeTypes ProbeType)
}

// NoopContainerProbeLifecycle is a no-op implementation of ContainerProbeLifecycle.
type NoopContainerProbeLifecycle struct{}

var _ ContainerProbeLifecycle = NoopContainerProbeLifecycle{}

func (NoopContainerProbeLifecycle) StartProbes(context.Context, *v1.Pod, *v1.Container, ContainerID, []string, time.Time) {
}

func (NoopContainerProbeLifecycle) StopProbes(ContainerID, ProbeType) {}
