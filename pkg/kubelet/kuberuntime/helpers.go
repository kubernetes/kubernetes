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
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/security/apparmor"
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

type containerStatusByCreated []*kubecontainer.Status

func (c containerStatusByCreated) Len() int           { return len(c) }
func (c containerStatusByCreated) Swap(i, j int)      { c[i], c[j] = c[j], c[i] }
func (c containerStatusByCreated) Less(i, j int) bool { return c[i].CreatedAt.After(c[j].CreatedAt) }

// toKubeContainerState converts runtimeapi.ContainerState to kubecontainer.State.
func toKubeContainerState(state runtimeapi.ContainerState) kubecontainer.State {
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

	klog.InfoS("Unknown protocol, defaulting to TCP", "protocol", protocol)
	return runtimeapi.Protocol_TCP
}

// toKubeContainer converts runtimeapi.Container to kubecontainer.Container.
func (m *kubeGenericRuntimeManager) toKubeContainer(c *runtimeapi.Container) (*kubecontainer.Container, error) {
	if c == nil || c.Id == "" || c.Image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	// Keep backwards compatibility to older runtimes, c.ImageId has been added in v1.30
	imageID := c.ImageRef
	if c.ImageId != "" {
		imageID = c.ImageId
	}

	annotatedInfo := getContainerInfoFromAnnotations(c.Annotations)
	return &kubecontainer.Container{
		ID:                   kubecontainer.ContainerID{Type: m.runtimeName, ID: c.Id},
		Name:                 c.GetMetadata().GetName(),
		ImageID:              imageID,
		ImageRef:             c.ImageRef,
		ImageRuntimeHandler:  c.Image.RuntimeHandler,
		Image:                c.Image.Image,
		Hash:                 annotatedInfo.Hash,
		HashWithoutResources: annotatedInfo.HashWithoutResources,
		State:                toKubeContainerState(c.State),
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
func (m *kubeGenericRuntimeManager) getImageUser(ctx context.Context, image string) (*int64, string, error) {
	resp, err := m.imageService.ImageStatus(ctx, &runtimeapi.ImageSpec{Image: image}, false)
	if err != nil {
		return nil, "", err
	}
	imageStatus := resp.GetImage()

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

// isInitContainerFailed returns true under the following conditions:
// 1. container has exited and exitcode is not zero.
// 2. container is in unknown state.
// 3. container gets OOMKilled.
func isInitContainerFailed(status *kubecontainer.Status) bool {
	// When oomkilled occurs, init container should be considered as a failure.
	if status.Reason == "OOMKilled" {
		return true
	}

	if status.State == kubecontainer.ContainerStateExited && status.ExitCode != 0 {
		return true
	}

	if status.State == kubecontainer.ContainerStateUnknown {
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

// logPathDelimiter is the delimiter used in the log path.
const logPathDelimiter = "_"

// buildContainerLogsPath builds log path for container relative to pod logs directory.
func buildContainerLogsPath(containerName string, restartCount int) string {
	return filepath.Join(containerName, fmt.Sprintf("%d.log", restartCount))
}

// BuildContainerLogsDirectory builds absolute log directory path for a container in pod.
func BuildContainerLogsDirectory(podLogsDir, podNamespace, podName string, podUID types.UID, containerName string) string {
	return filepath.Join(BuildPodLogsDirectory(podLogsDir, podNamespace, podName, podUID), containerName)
}

// BuildPodLogsDirectory builds absolute log directory path for a pod sandbox.
func BuildPodLogsDirectory(podLogsDir, podNamespace, podName string, podUID types.UID) string {
	return filepath.Join(podLogsDir, strings.Join([]string{podNamespace, podName,
		string(podUID)}, logPathDelimiter))
}

// parsePodUIDFromLogsDirectory parses pod logs directory name and returns the pod UID.
// It supports both the old pod log directory /var/log/pods/UID, and the new pod log
// directory /var/log/pods/NAMESPACE_NAME_UID.
func parsePodUIDFromLogsDirectory(name string) types.UID {
	parts := strings.Split(name, logPathDelimiter)
	return types.UID(parts[len(parts)-1])
}

// toKubeRuntimeStatus converts the runtimeapi.RuntimeStatus to kubecontainer.RuntimeStatus.
func toKubeRuntimeStatus(status *runtimeapi.RuntimeStatus, handlers []*runtimeapi.RuntimeHandler) *kubecontainer.RuntimeStatus {
	conditions := []kubecontainer.RuntimeCondition{}
	for _, c := range status.GetConditions() {
		conditions = append(conditions, kubecontainer.RuntimeCondition{
			Type:    kubecontainer.RuntimeConditionType(c.Type),
			Status:  c.Status,
			Reason:  c.Reason,
			Message: c.Message,
		})
	}
	retHandlers := make([]kubecontainer.RuntimeHandler, len(handlers))
	for i, h := range handlers {
		supportsRRO := false
		supportsUserns := false
		if h.Features != nil {
			supportsRRO = h.Features.RecursiveReadOnlyMounts
			supportsUserns = h.Features.UserNamespaces
		}
		retHandlers[i] = kubecontainer.RuntimeHandler{
			Name:                            h.Name,
			SupportsRecursiveReadOnlyMounts: supportsRRO,
			SupportsUserNamespaces:          supportsUserns,
		}
	}
	return &kubecontainer.RuntimeStatus{Conditions: conditions, Handlers: retHandlers}
}

func fieldSeccompProfile(scmp *v1.SeccompProfile, profileRootPath string, fallbackToRuntimeDefault bool) (*runtimeapi.SecurityProfile, error) {
	if scmp == nil {
		if fallbackToRuntimeDefault {
			return &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
			}, nil
		}
		return &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_Unconfined,
		}, nil
	}
	if scmp.Type == v1.SeccompProfileTypeRuntimeDefault {
		return &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
		}, nil
	}
	if scmp.Type == v1.SeccompProfileTypeLocalhost {
		if scmp.LocalhostProfile != nil && len(*scmp.LocalhostProfile) > 0 {
			fname := filepath.Join(profileRootPath, *scmp.LocalhostProfile)
			return &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: fname,
			}, nil
		} else {
			return nil, fmt.Errorf("localhostProfile must be set if seccompProfile type is Localhost.")
		}
	}
	return &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_Unconfined,
	}, nil
}

func (m *kubeGenericRuntimeManager) getSeccompProfile(annotations map[string]string, containerName string,
	podSecContext *v1.PodSecurityContext, containerSecContext *v1.SecurityContext, fallbackToRuntimeDefault bool) (*runtimeapi.SecurityProfile, error) {
	// container fields are applied first
	if containerSecContext != nil && containerSecContext.SeccompProfile != nil {
		return fieldSeccompProfile(containerSecContext.SeccompProfile, m.seccompProfileRoot, fallbackToRuntimeDefault)
	}

	// when container seccomp is not defined, try to apply from pod field
	if podSecContext != nil && podSecContext.SeccompProfile != nil {
		return fieldSeccompProfile(podSecContext.SeccompProfile, m.seccompProfileRoot, fallbackToRuntimeDefault)
	}

	if fallbackToRuntimeDefault {
		return &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
		}, nil
	}

	return &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_Unconfined,
	}, nil
}

func getAppArmorProfile(pod *v1.Pod, container *v1.Container) (*runtimeapi.SecurityProfile, string, error) {
	profile := apparmor.GetProfile(pod, container)
	if profile == nil {
		return nil, "", nil
	}

	var (
		securityProfile   *runtimeapi.SecurityProfile
		deprecatedProfile string // Deprecated apparmor profile format, still provided for backwards compatibility with older runtimes.
	)

	switch profile.Type {
	case v1.AppArmorProfileTypeRuntimeDefault:
		securityProfile = &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
		}
		deprecatedProfile = v1.DeprecatedAppArmorBetaProfileRuntimeDefault

	case v1.AppArmorProfileTypeUnconfined:
		securityProfile = &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_Unconfined,
		}
		deprecatedProfile = v1.DeprecatedAppArmorBetaProfileNameUnconfined

	case v1.AppArmorProfileTypeLocalhost:
		if profile.LocalhostProfile == nil {
			return nil, "", errors.New("missing localhost apparmor profile name")
		}
		securityProfile = &runtimeapi.SecurityProfile{
			ProfileType:  runtimeapi.SecurityProfile_Localhost,
			LocalhostRef: *profile.LocalhostProfile,
		}
		deprecatedProfile = v1.DeprecatedAppArmorBetaProfileNamePrefix + *profile.LocalhostProfile

	default:
		// Shouldn't happen.
		return nil, "", fmt.Errorf("unknown apparmor profile type: %q", profile.Type)
	}

	return securityProfile, deprecatedProfile, nil
}
