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
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/probe"
	"time"
)

// grpcProbeRunner holds state and logic for running GRPC probes.
type grpcProbeRunner struct {
	host    string
	service string
	port    int
	timeout time.Duration
}

// newGRPCProbeRunner returns grpcProbeRunner which implements probeRunner.
func newGRPCProbeRunner() *grpcProbeRunner {
	return &grpcProbeRunner{}
}

func (gp *grpcProbeRunner) sync(container v1.Container, status v1.PodStatus, probeType probeType) error {
	if gp.host == status.PodIP {
		// no sync required, podIP hasn't changed.
		return nil
	}

	// no need to handle error here, factory will handle this case.
	probeSpec, _ := getProbeSpecFromContainer(container, probeType)

	timeout := time.Duration(probeSpec.TimeoutSeconds) * time.Second

	host := status.PodIP
	service := ""
	if probeSpec.GRPC.Service != nil {
		service = *probeSpec.GRPC.Service
	}

	// persist
	gp.host = host
	gp.service = service
	gp.port = int(probeSpec.GRPC.Port)
	gp.timeout = timeout
	return nil
}

func (gp *grpcProbeRunner) run(_ context.Context, container v1.Container, status v1.PodStatus, probeType probeType, prober *prober) (probe.Result, string, error) {
	err := gp.sync(container, status, probeType)
	if err != nil {
		return probe.Unknown, "", err
	}

	klog.V(4).InfoS("GRPC-Probe", "host", gp.host, "service", gp.service, "port", gp.port, "timeout", gp.timeout)
	return prober.grpc.Probe(gp.host, gp.service, gp.port, gp.timeout)
}
