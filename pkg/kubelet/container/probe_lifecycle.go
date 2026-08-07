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

	// StopLivenessAndStartupProbes stops the liveness and startup probes of a
	// container that is about to be killed, aborting any probe currently
	// executing. It is called before the PreStop hook, so that a container on
	// its way out is not killed again for failing a probe and no exec probe is
	// left running inside it as it is torn down.
	//
	// Readiness deliberately keeps running: a container that is shutting down
	// should be taken out of service, which is what its readiness probe failing
	// does.
	StopLivenessAndStartupProbes(containerID ContainerID)

	// StopProbes stops all remaining probes for a container instance and drops
	// its cached results. It is called once the container has stopped.
	StopProbes(containerID ContainerID)
}
