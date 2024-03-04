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

// tcpProbeRunner holds state and logic for running TCP probes.
type tcpProbeRunner struct {
	host    string
	port    int
	timeout time.Duration
}

// newTCPProbeRunner returns tcpProbeRunner which implements probeRunner.
func newTCPProbeRunner() *tcpProbeRunner {
	return &tcpProbeRunner{}
}

func (tp *tcpProbeRunner) sync(container v1.Container, status v1.PodStatus, probeType probeType) error {
	if tp.host == status.PodIP {
		// no sync required, podIP hasn't changed.
		return nil
	}

	// no need to handle error here, factory will handle this case.
	probeSpec, _ := getProbeSpecFromContainer(container, probeType)

	timeout := time.Duration(probeSpec.TimeoutSeconds) * time.Second
	port, err := probe.ResolveContainerPort(probeSpec.TCPSocket.Port, &container)
	if err != nil {
		return err
	}

	host := probeSpec.TCPSocket.Host
	if host == "" {
		host = status.PodIP
	}

	// persist
	tp.host = host
	tp.port = port
	tp.timeout = timeout
	return err
}

func (tp *tcpProbeRunner) run(_ context.Context, container v1.Container, status v1.PodStatus, probeType probeType, prober *prober) (probe.Result, string, error) {
	err := tp.sync(container, status, probeType)
	if err != nil {
		return probe.Unknown, "", err
	}

	klog.V(4).InfoS("TCP-Probe", "host", tp.host, "port", tp.port, "timeout", tp.timeout)
	return prober.tcp.Probe(tp.host, tp.port, tp.timeout)
}
