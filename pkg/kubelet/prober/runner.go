/*
Copyright 2023 The Kubernetes Authors.

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

package prober

import (
	"context"
	"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/probe"
)

// probeRunner is an interface for running probes, objects implementing
// probeRunner encapsulates the logic of running probes.
type probeRunner interface {

	// sync ensures objects required for running probe are in sync, with probe
	// specification and latest pod status (podIP can mutate for running pod).
	sync(v1.Container, v1.PodStatus, probeType) error

	// run holds the logic of running probe, run takes prober as a dependency
	// to run the probe on http.Prober, tcp.Prober, grpc.Prober or exec.Prober.
	run(context.Context, v1.Container, v1.PodStatus, probeType, *prober) (probe.Result, string, error)
}

// newProbeRunner is a factory which returns concrete implementation of probeRunner
// depending on the probe handler (tcp, http, grpc and exec).
func newProbeRunner(container v1.Container, probeType probeType) probeRunner {
	probeSpec, err := getProbeSpecFromContainer(container, probeType)
	if err != nil {
		// return dummyProbeRunner which on run returns failure status and error.
		return newDummyProbeRunner(true, false, false)
	}

	// return dummyProbeRunner which on run returns successful status and no error.
	if probeSpec == nil {
		return newDummyProbeRunner(false, true, false)
	}

	// return tcpProbeRunner for running TCP probes.
	if probeSpec.TCPSocket != nil {
		return newTCPProbeRunner()
	}

	// return httpProbeRunner for running HTTP probes.
	if probeSpec.HTTPGet != nil {
		return newHTTPProbeRunner()
	}

	// return grpcProbeRunner for running GRPC probes.
	if probeSpec.GRPC != nil {
		return newGRPCProbeRunner()
	}

	// return execProbeRunner for running Exec probes.
	if probeSpec.Exec != nil {
		return newExecProbeRunner()
	}

	// return dummyProbeRunner which on run returns unknown status and error.
	return newDummyProbeRunner(false, false, true)
}

func getProbeSpecFromContainer(container v1.Container, probeType probeType) (*v1.Probe, error) {
	var probeSpec *v1.Probe
	switch probeType {
	case startup:
		probeSpec = container.StartupProbe
	case liveness:
		probeSpec = container.LivenessProbe
	case readiness:
		probeSpec = container.ReadinessProbe
	default:
		return nil, fmt.Errorf("unknown probe type: %q", probeType)
	}
	return probeSpec, nil
}
