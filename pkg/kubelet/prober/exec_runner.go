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
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/probe"
	"time"
)

// execProbeRunner holds state and logic for running Exec probes.
type execProbeRunner struct {
	synced      bool
	containerID kubecontainer.ContainerID
	command     []string
	timeout     time.Duration
}

// newExecProbeRunner returns execProbeRunner which implements probeRunner.
func newExecProbeRunner() *execProbeRunner {
	return &execProbeRunner{}
}

func (ep *execProbeRunner) sync(container v1.Container, status v1.PodStatus, probeType probeType) error {
	// we only need to sync execProberRunner once as it doesn't depend on podIP and all other
	// required objects are immutable, not using sync.Once here to avoid un-necessary mutexes.
	if ep.synced {
		return nil
	}

	// no need to handle error here, factory will handle this case.
	probeSpec, _ := getProbeSpecFromContainer(container, probeType)

	command := kubecontainer.ExpandContainerCommandOnlyStatic(probeSpec.Exec.Command, container.Env)
	timeout := time.Duration(probeSpec.TimeoutSeconds) * time.Second

	c, _ := podutil.GetContainerStatus(status.ContainerStatuses, container.Name)
	containerID := kubecontainer.ParseContainerID(c.ContainerID)

	// persist
	ep.synced = true
	ep.command = command
	ep.timeout = timeout
	ep.containerID = containerID
	return nil
}

func (ep *execProbeRunner) run(ctx context.Context, container v1.Container, status v1.PodStatus, probeType probeType, prober *prober) (probe.Result, string, error) {
	err := ep.sync(container, status, probeType)
	if err != nil {
		return probe.Unknown, "", err
	}

	klog.V(4).InfoS("Exec-Probe", "execCommand", ep.command, "containerID", ep.containerID.String())
	return prober.exec.Probe(&execInContainer{run: func() ([]byte, error) {
		return prober.runner.RunInContainer(ctx, ep.containerID, ep.command, ep.timeout)
	}})
}
