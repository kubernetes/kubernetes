/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	quotaPeriod    = 100 * minQuotaPeriod
	minQuotaPeriod = 1000
)

type podsByID []*kubecontainer.Pod

func (b podsByID) Len() int           { return len(b) }
func (b podsByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b podsByID) Less(i, j int) bool { return b[i].ID < b[j].ID }

type containersByID []*kubecontainer.Container

func (b containersByID) Len() int           { return len(b) }
func (b containersByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b containersByID) Less(i, j int) bool { return b[i].ID.ID < b[j].ID.ID }

// toKubeContainerState converts runtimeApi.ContainerState to kubecontainer.ContainerState.
func toKubeContainerState(state runtimeApi.ContainerState) kubecontainer.ContainerState {
	switch state {
	case runtimeApi.ContainerState_CREATED:
		return kubecontainer.ContainerStateCreated
	case runtimeApi.ContainerState_RUNNING:
		return kubecontainer.ContainerStateRunning
	case runtimeApi.ContainerState_EXITED:
		return kubecontainer.ContainerStateExited
	case runtimeApi.ContainerState_UNKNOWN:
		return kubecontainer.ContainerStateUnknown
	}

	return kubecontainer.ContainerStateUnknown
}

// toRuntimeProtocol converts api.Protocol to runtimeApi.Protocol.
func toRuntimeProtocol(protocol api.Protocol) runtimeApi.Protocol {
	switch protocol {
	case api.ProtocolTCP:
		return runtimeApi.Protocol_TCP
	case api.ProtocolUDP:
		return runtimeApi.Protocol_UDP
	}

	glog.Warningf("Unknown protocol %q: defaulting to TCP", protocol)
	return runtimeApi.Protocol_TCP
}

// toKubeContainer converts runtimeApi.Container to kubecontainer.Container.
func (m *kubeGenericRuntimeManager) toKubeContainer(c *runtimeApi.Container) (*kubecontainer.Container, error) {
	if c == nil || c.Id == nil || c.Image == nil || c.State == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	labeledInfo := getContainerInfoFromLabels(c.Labels)
	annotatedInfo := getContainerInfoFromAnnotations(c.Annotations)
	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: m.runtimeName, ID: c.GetId()},
		Name:  labeledInfo.ContainerName,
		Image: c.Image.GetImage(),
		Hash:  annotatedInfo.Hash,
		State: toKubeContainerState(c.GetState()),
	}, nil
}

// milliCPUToShares converts milliCPU to CPU shares
func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// Return 2 here to really match kernel default for zero milliCPU.
		return minShares
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

// milliCPUToQuota converts milliCPU to CFS quota and period values
func milliCPUToQuota(milliCPU int64) (quota int64, period int64) {
	// CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts
	if milliCPU == 0 {
		return
	}

	// we set the period to 100ms by default
	period = quotaPeriod

	// we then convert your milliCPU to a value normalized over a period
	quota = (milliCPU * quotaPeriod) / milliCPUToCPU

	// quota needs to be a minimum of 1ms.
	if quota < minQuotaPeriod {
		quota = minQuotaPeriod
	}

	return
}
