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
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
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
func (p podSandboxByCreated) Less(i, j int) bool { return p[i].CreatedAt > p[j].CreatedAt }

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
	case v1.ProtocolSCTP:
		return runtimeapi.Protocol_SCTP
	}

	glog.Warningf("Unknown protocol %q: defaulting to TCP", protocol)
	return runtimeapi.Protocol_TCP
}

// toKubeContainer converts runtimeapi.Container to kubecontainer.Container.
func (m *kubeGenericRuntimeManager) toKubeContainer(c *runtimeapi.Container) (*kubecontainer.Container, error) {
	if c == nil || c.Id == "" || c.Image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	annotatedInfo := getContainerInfoFromAnnotations(c.Annotations)
	return &kubecontainer.Container{
		ID:      kubecontainer.ContainerID{Type: m.runtimeName, ID: c.Id},
		Name:    c.GetMetadata().GetName(),
		ImageID: c.ImageRef,
		Image:   c.Image.Image,
		Hash:    annotatedInfo.Hash,
		State:   toKubeContainerState(c.State),
	}, nil
}

// sandboxToKubeContainer converts runtimeapi.PodSandbox to kubecontainer.Container.
// This is only needed because we need to return sandboxes as if they were
// kubecontainer.Containers to avoid substantial changes to PLEG.
// TODO: Remove this once it becomes obsolete.
func (m *kubeGenericRuntimeManager) sandboxToKubeContainer(s *runtimeapi.PodSandbox) (*kubecontainer.Container, error) {
	if s == nil || s.Id == "" {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: m.runtimeName, ID: s.Id},
		State: kubecontainer.SandboxToContainerState(s.State),
	}, nil
}

// getImageUser gets uid or user name that will run the command(s) from image. The function
// guarantees that only one of them is set.
func (m *kubeGenericRuntimeManager) getImageUser(image string) (*int64, string, error) {
	imageStatus, err := m.imageService.ImageStatus(&runtimeapi.ImageSpec{Image: image})
	if err != nil {
		return nil, "", err
	}

	if imageStatus != nil {
		if imageStatus.Uid != nil {
			return &imageStatus.GetUid().Value, "", nil
		}

		if imageStatus.Username != "" {
			return nil, imageStatus.Username, nil
		}
	}

	// If non of them is set, treat it as root.
	return new(int64), "", nil
}

// isContainerFailed returns true if container has exited and exitcode is not zero.
func isContainerFailed(status *kubecontainer.ContainerStatus) bool {
	if status.State == kubecontainer.ContainerStateExited && status.ExitCode != 0 {
		return true
	}

	return false
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
	return filepath.Join(containerName, fmt.Sprintf("%d.log", restartCount))
}

// buildFullContainerLogsPath builds absolute log path for container.
func buildFullContainerLogsPath(podUID types.UID, containerName string, restartCount int) string {
	return filepath.Join(buildPodLogsDirectory(podUID), buildContainerLogsPath(containerName, restartCount))
}

// BuildContainerLogsDirectory builds absolute log directory path for a container in pod.
func BuildContainerLogsDirectory(podUID types.UID, containerName string) string {
	return filepath.Join(buildPodLogsDirectory(podUID), containerName)
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
			Type:    kubecontainer.RuntimeConditionType(c.Type),
			Status:  c.Status,
			Reason:  c.Reason,
			Message: c.Message,
		})
	}
	return &kubecontainer.RuntimeStatus{Conditions: conditions}
}

// getSeccompProfileFromAnnotations gets seccomp profile from annotations.
// It gets pod's profile if containerName is empty.
func (m *kubeGenericRuntimeManager) getSeccompProfileFromAnnotations(annotations map[string]string, containerName string) string {
	// try the pod profile.
	profile, profileOK := annotations[v1.SeccompPodAnnotationKey]
	if containerName != "" {
		// try the container profile.
		cProfile, cProfileOK := annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName]
		if cProfileOK {
			profile = cProfile
			profileOK = cProfileOK
		}
	}

	if !profileOK {
		return ""
	}

	if strings.HasPrefix(profile, "localhost/") {
		name := strings.TrimPrefix(profile, "localhost/")
		fname := filepath.Join(m.seccompProfileRoot, filepath.FromSlash(name))
		return "localhost/" + fname
	}

	return profile
}

func ipcNamespaceForPod(pod *v1.Pod) runtimeapi.NamespaceMode {
	if pod != nil && pod.Spec.HostIPC {
		return runtimeapi.NamespaceMode_NODE
	}
	return runtimeapi.NamespaceMode_POD
}

func networkNamespaceForPod(pod *v1.Pod) runtimeapi.NamespaceMode {
	if pod != nil && pod.Spec.HostNetwork {
		return runtimeapi.NamespaceMode_NODE
	}
	return runtimeapi.NamespaceMode_POD
}

func pidNamespaceForPod(pod *v1.Pod) runtimeapi.NamespaceMode {
	if pod != nil {
		if pod.Spec.HostPID {
			return runtimeapi.NamespaceMode_NODE
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.PodShareProcessNamespace) && pod.Spec.ShareProcessNamespace != nil && *pod.Spec.ShareProcessNamespace {
			return runtimeapi.NamespaceMode_POD
		}
	}
	// Note that PID does not default to the zero value for v1.Pod
	return runtimeapi.NamespaceMode_CONTAINER
}

// namespacesForPod returns the runtimeapi.NamespaceOption for a given pod.
// An empty or nil pod can be used to get the namespace defaults for v1.Pod.
func namespacesForPod(pod *v1.Pod) *runtimeapi.NamespaceOption {
	return &runtimeapi.NamespaceOption{
		Ipc:     ipcNamespaceForPod(pod),
		Network: networkNamespaceForPod(pod),
		Pid:     pidNamespaceForPod(pod),
	}
}
