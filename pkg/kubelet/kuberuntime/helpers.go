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
	"path/filepath"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
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

var (
	// The default dns opt strings
	defaultDNSOptions = []string{"ndots:5"}
)

type podsByID []*kubecontainer.Pod

func (b podsByID) Len() int           { return len(b) }
func (b podsByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b podsByID) Less(i, j int) bool { return b[i].ID < b[j].ID }

type containersByID []*kubecontainer.Container

func (b containersByID) Len() int           { return len(b) }
func (b containersByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b containersByID) Less(i, j int) bool { return b[i].ID.ID < b[j].ID.ID }

// Newest first.
type podSandboxByCreated []*runtimeapi.PodSandbox

func (p podSandboxByCreated) Len() int           { return len(p) }
func (p podSandboxByCreated) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p podSandboxByCreated) Less(i, j int) bool { return p[i].GetCreatedAt() > p[j].GetCreatedAt() }

type containerStatusByCreated []*kubecontainer.ContainerStatus

func (c containerStatusByCreated) Len() int           { return len(c) }
func (c containerStatusByCreated) Swap(i, j int)      { c[i], c[j] = c[j], c[i] }
func (c containerStatusByCreated) Less(i, j int) bool { return c[i].CreatedAt.After(c[j].CreatedAt) }

// toKubeContainerState converts runtimeapi.ContainerState to kubecontainer.ContainerState.
func toKubeContainerState(state runtimeapi.ContainerState) kubecontainer.ContainerState {
	switch state {
	case runtimeapi.ContainerState_CONTAINER_CREATED:
		return kubecontainer.ContainerStateCreated
	case runtimeapi.ContainerState_CONTAINER_RUNNING:
		return kubecontainer.ContainerStateRunning
	case runtimeapi.ContainerState_CONTAINER_EXITED:
		return kubecontainer.ContainerStateExited
	case runtimeapi.ContainerState_CONTAINER_UNKNOWN:
		return kubecontainer.ContainerStateUnknown
	}

	return kubecontainer.ContainerStateUnknown
}

// toRuntimeProtocol converts v1.Protocol to runtimeapi.Protocol.
func toRuntimeProtocol(protocol v1.Protocol) runtimeapi.Protocol {
	switch protocol {
	case v1.ProtocolTCP:
		return runtimeapi.Protocol_TCP
	case v1.ProtocolUDP:
		return runtimeapi.Protocol_UDP
	}

	glog.Warningf("Unknown protocol %q: defaulting to TCP", protocol)
	return runtimeapi.Protocol_TCP
}

// toKubeContainer converts runtimeapi.Container to kubecontainer.Container.
func (m *kubeGenericRuntimeManager) toKubeContainer(c *runtimeapi.Container) (*kubecontainer.Container, error) {
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

// sandboxToKubeContainer converts runtimeapi.PodSandbox to kubecontainer.Container.
// This is only needed because we need to return sandboxes as if they were
// kubecontainer.Containers to avoid substantial changes to PLEG.
// TODO: Remove this once it becomes obsolete.
func (m *kubeGenericRuntimeManager) sandboxToKubeContainer(s *runtimeapi.PodSandbox) (*kubecontainer.Container, error) {
	if s == nil || s.Id == nil || s.State == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: m.runtimeName, ID: s.GetId()},
		State: kubecontainer.SandboxToContainerState(s.GetState()),
	}, nil
}

// getImageUser gets uid or user name that will run the command(s) from image. The function
// guarantees that only one of them is set.
func (m *kubeGenericRuntimeManager) getImageUser(image string) (*int64, *string, error) {
	imageStatus, err := m.imageService.ImageStatus(&runtimeapi.ImageSpec{Image: &image})
	if err != nil {
		return nil, nil, err
	}

	if imageStatus != nil && imageStatus.Uid != nil {
		// If uid is set, return uid.
		return imageStatus.Uid, nil, nil
	}
	if imageStatus != nil && imageStatus.Username != nil {
		// If uid is not set, but user name is set, return user name.
		return nil, imageStatus.Username, nil
	}
	// If non of them is set, treat it as root.
	return new(int64), nil, nil
}

// isContainerFailed returns true if the container has exited and exitcode is not zero.
func isContainerFailed(status *kubecontainer.ContainerStatus) bool {
	switch status.State {
	case kubecontainer.ContainerStateUnknown:
		return true
	case kubecontainer.ContainerStateExited:
		return status.ExitCode != 0
	}
	return false
}

// isContainerRunning returns true if the container is running.
func isContainerRunning(status *kubecontainer.ContainerStatus) bool {
	return status.State == kubecontainer.ContainerStateRunning
}

// isContainerActive returns true if the container is created or running.
func isContainerActive(status *kubecontainer.ContainerStatus) bool {
	return status.State == kubecontainer.ContainerStateRunning || status.State == kubecontainer.ContainerStateCreated
}

// isContainerExited returns true if the container is exited.
func isContainerExited(status *kubecontainer.ContainerStatus) bool {
	return status.State == kubecontainer.ContainerStateExited
}

// foundInitContainerSpec returns the init container's spec if found.
func foundInitContainerSpec(pod *v1.Pod, status *kubecontainer.ContainerStatus) *v1.Container {
	for _, c := range pod.Spec.InitContainers {
		if c.Name == status.Name {
			return &c
		}
	}
	return nil
}

// containerChanged returns true if the container's spec has changed.
func containerChanged(container *v1.Container, status *kubecontainer.ContainerStatus) bool {
	return kubecontainer.HashContainer(container) != status.Hash
}

// initContainersChanged returns true if any init container's spec has been changed.
// It's done by comparing the latest init container hash with the old one.
//
// TODO(yifan): Implement this by storing the init container hash in the pod annotation.
func initContainersChanged(pod *v1.Pod, sandboxStatus *runtimeapi.PodSandboxStatus) bool {
	return false
}

// containerUnhealthy returns true if the container's liveness probing result
// is found but not success.
func containerUnhealthy(manager proberesults.Manager, status *kubecontainer.ContainerStatus) bool {
	liveness, found := manager.Get(status.ID)
	return found && liveness != proberesults.Success
}

// findContainerSpecByName returns the init or app container spec that has
// the given name.
func findContainerSpecByName(name string, pod *v1.Pod) *v1.Container {
	for _, c := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		if name == c.Name {
			return &c
		}
	}
	return nil
}

// isInBackOff returns true and the backoff time if the container is still in back-off.
func isInBackOff(pod *v1.Pod, container *v1.Container, finishedAt time.Time, backoff *flowcontrol.Backoff) (bool, time.Duration) {
	glog.V(4).Infof("Checking backoff for container %q in pod %q", container.Name, format.Pod(pod))
	// Use the finished time of the latest exited container as the start point to calculate whether to do back-off.
	var backoffTime time.Duration
	// backoff requires a unique key to identify the container.
	key := getStableKey(pod, container)
	if backoff.IsInBackOffSince(key, finishedAt) {
		backoffTime = backoff.Get(key)
		glog.V(4).Infof("The container %q of pod %q is in back-off, will restart in %v", container.Name, format.Pod(pod), backoffTime)
		return true, backoffTime
	}

	backoff.Next(key, finishedAt)
	return false, backoffTime
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

// getStableKey generates a key (string) to uniquely identify a
// (pod, container) tuple. The key should include the content of the
// container, so that any change to the container generates a new key.
func getStableKey(pod *v1.Pod, container *v1.Container) string {
	hash := strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	return fmt.Sprintf("%s_%s_%s_%s_%s", pod.Name, pod.Namespace, string(pod.UID), container.Name, hash)
}

// buildContainerLogsPath builds log path for container relative to pod logs directory.
func buildContainerLogsPath(containerName string, restartCount int) string {
	return fmt.Sprintf("%s_%d.log", containerName, restartCount)
}

// buildFullContainerLogsPath builds absolute log path for container.
func buildFullContainerLogsPath(podUID types.UID, containerName string, restartCount int) string {
	return filepath.Join(buildPodLogsDirectory(podUID), buildContainerLogsPath(containerName, restartCount))
}

// buildPodLogsDirectory builds absolute log directory path for a pod sandbox.
func buildPodLogsDirectory(podUID types.UID) string {
	return filepath.Join(podLogsRootDirectory, string(podUID))
}

// toKubeRuntimeStatus converts the runtimeapi.RuntimeStatus to kubecontainer.RuntimeStatus.
func toKubeRuntimeStatus(status *runtimeapi.RuntimeStatus) *kubecontainer.RuntimeStatus {
	conditions := []kubecontainer.RuntimeCondition{}
	for _, c := range status.GetConditions() {
		conditions = append(conditions, kubecontainer.RuntimeCondition{
			Type:    kubecontainer.RuntimeConditionType(c.GetType()),
			Status:  c.GetStatus(),
			Reason:  c.GetReason(),
			Message: c.GetMessage(),
		})
	}
	return &kubecontainer.RuntimeStatus{Conditions: conditions}
}
