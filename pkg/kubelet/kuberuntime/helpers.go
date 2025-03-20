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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
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
		ID:                  kubecontainer.ContainerID{Type: m.runtimeName, ID: c.Id},
		Name:                c.GetMetadata().GetName(),
		ImageID:             imageID,
		ImageRef:            c.ImageRef,
		ImageRuntimeHandler: c.Image.RuntimeHandler,
		Image:               c.Image.Image,
		Hash:                annotatedInfo.Hash,
		State:               toKubeContainerState(c.State),
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

// GetStableKey generates a key (string) to uniquely identify a
// (pod, container) tuple. The key should include the content of the
// container, so that any change to the container generates a new key.
func GetStableKey(pod *v1.Pod, container *v1.Container) string {
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
func toKubeRuntimeStatus(status *runtimeapi.RuntimeStatus, handlers []*runtimeapi.RuntimeHandler, features *runtimeapi.RuntimeFeatures) *kubecontainer.RuntimeStatus {
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
	var retFeatures *kubecontainer.RuntimeFeatures
	if features != nil {
		retFeatures = &kubecontainer.RuntimeFeatures{
			SupplementalGroupsPolicy: features.SupplementalGroupsPolicy,
		}
	}
	return &kubecontainer.RuntimeStatus{Conditions: conditions, Handlers: retHandlers, Features: retFeatures}
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

func mergeResourceConfig(source, update *cm.ResourceConfig) *cm.ResourceConfig {
	if source == nil {
		return update
	}
	if update == nil {
		return source
	}

	merged := *source

	if update.Memory != nil {
		merged.Memory = update.Memory
	}
	if update.CPUSet.Size() > 0 {
		merged.CPUSet = update.CPUSet
	}
	if update.CPUShares != nil {
		merged.CPUShares = update.CPUShares
	}
	if update.CPUQuota != nil {
		merged.CPUQuota = update.CPUQuota
	}
	if update.CPUPeriod != nil {
		merged.CPUPeriod = update.CPUPeriod
	}
	if update.PidsLimit != nil {
		merged.PidsLimit = update.PidsLimit
	}

	if update.HugePageLimit != nil {
		if merged.HugePageLimit == nil {
			merged.HugePageLimit = make(map[int64]int64)
		}
		for k, v := range update.HugePageLimit {
			merged.HugePageLimit[k] = v
		}
	}

	if update.Unified != nil {
		if merged.Unified == nil {
			merged.Unified = make(map[string]string)
		}
		for k, v := range update.Unified {
			merged.Unified[k] = v
		}
	}

	return &merged
}

func convertResourceConfigToLinuxContainerResources(rc *cm.ResourceConfig) *runtimeapi.LinuxContainerResources {
	if rc == nil {
		return nil
	}

	lcr := &runtimeapi.LinuxContainerResources{}

	if rc.CPUPeriod != nil {
		lcr.CpuPeriod = int64(*rc.CPUPeriod)
	}
	if rc.CPUQuota != nil {
		lcr.CpuQuota = *rc.CPUQuota
	}
	if rc.CPUShares != nil {
		lcr.CpuShares = int64(*rc.CPUShares)
	}
	if rc.Memory != nil {
		lcr.MemoryLimitInBytes = *rc.Memory
	}
	if rc.CPUSet.Size() > 0 {
		lcr.CpusetCpus = rc.CPUSet.String()
	}

	if rc.Unified != nil {
		lcr.Unified = make(map[string]string, len(rc.Unified))
		for k, v := range rc.Unified {
			lcr.Unified[k] = v
		}
	}

	return lcr
}

func buildContainerConfigLifecycle(container *v1.Container) (lifecycle *runtimeapi.Lifecycle) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ContainerStopSignals) {
		if container.Lifecycle != nil && container.Lifecycle.StopSignal != nil {
			var signalValue runtimeapi.Signal
			signalStr := string(*container.Lifecycle.StopSignal)

			signalValue = runtimeapi.Signal_RUNTIME_DEFAULT

			switch signalStr {
			case "SIGABRT":
				signalValue = runtimeapi.Signal_SIGABRT
			case "SIGALRM":
				signalValue = runtimeapi.Signal_SIGALRM
			case "SIGBUS":
				signalValue = runtimeapi.Signal_SIGBUS
			case "SIGCHLD":
				signalValue = runtimeapi.Signal_SIGCHLD
			case "SIGCLD":
				signalValue = runtimeapi.Signal_SIGCLD
			case "SIGCONT":
				signalValue = runtimeapi.Signal_SIGCONT
			case "SIGFPE":
				signalValue = runtimeapi.Signal_SIGFPE
			case "SIGHUP":
				signalValue = runtimeapi.Signal_SIGHUP
			case "SIGILL":
				signalValue = runtimeapi.Signal_SIGILL
			case "SIGINT":
				signalValue = runtimeapi.Signal_SIGINT
			case "SIGIO":
				signalValue = runtimeapi.Signal_SIGIO
			case "SIGIOT":
				signalValue = runtimeapi.Signal_SIGIOT
			case "SIGKILL":
				signalValue = runtimeapi.Signal_SIGKILL
			case "SIGPIPE":
				signalValue = runtimeapi.Signal_SIGPIPE
			case "SIGPOLL":
				signalValue = runtimeapi.Signal_SIGPOLL
			case "SIGPROF":
				signalValue = runtimeapi.Signal_SIGPROF
			case "SIGPWR":
				signalValue = runtimeapi.Signal_SIGPWR
			case "SIGQUIT":
				signalValue = runtimeapi.Signal_SIGQUIT
			case "SIGSEGV":
				signalValue = runtimeapi.Signal_SIGSEGV
			case "SIGSTKFLT":
				signalValue = runtimeapi.Signal_SIGSTKFLT
			case "SIGSTOP":
				signalValue = runtimeapi.Signal_SIGSTOP
			case "SIGSYS":
				signalValue = runtimeapi.Signal_SIGSYS
			case "SIGTERM":
				signalValue = runtimeapi.Signal_SIGTERM
			case "SIGTRAP":
				signalValue = runtimeapi.Signal_SIGTRAP
			case "SIGTSTP":
				signalValue = runtimeapi.Signal_SIGTSTP
			case "SIGTTIN":
				signalValue = runtimeapi.Signal_SIGTTIN
			case "SIGTTOU":
				signalValue = runtimeapi.Signal_SIGTTOU
			case "SIGURG":
				signalValue = runtimeapi.Signal_SIGURG
			case "SIGUSR1":
				signalValue = runtimeapi.Signal_SIGUSR1
			case "SIGUSR2":
				signalValue = runtimeapi.Signal_SIGUSR2
			case "SIGVTALRM":
				signalValue = runtimeapi.Signal_SIGVTALRM
			case "SIGWINCH":
				signalValue = runtimeapi.Signal_SIGWINCH
			case "SIGXCPU":
				signalValue = runtimeapi.Signal_SIGXCPU
			case "SIGXFSZ":
				signalValue = runtimeapi.Signal_SIGXFSZ
			case "SIGRTMIN":
				signalValue = runtimeapi.Signal_SIGRTMIN
			case "SIGRTMIN+1":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS1
			case "SIGRTMIN+2":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS2
			case "SIGRTMIN+3":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS3
			case "SIGRTMIN+4":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS4
			case "SIGRTMIN+5":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS5
			case "SIGRTMIN+6":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS6
			case "SIGRTMIN+7":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS7
			case "SIGRTMIN+8":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS8
			case "SIGRTMIN+9":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS9
			case "SIGRTMIN+10":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS10
			case "SIGRTMIN+11":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS11
			case "SIGRTMIN+12":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS12
			case "SIGRTMIN+13":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS13
			case "SIGRTMIN+14":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS14
			case "SIGRTMIN+15":
				signalValue = runtimeapi.Signal_SIGRTMINPLUS15
			case "SIGRTMAX-14":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS14
			case "SIGRTMAX-13":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS13
			case "SIGRTMAX-12":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS12
			case "SIGRTMAX-11":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS11
			case "SIGRTMAX-10":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS10
			case "SIGRTMAX-9":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS9
			case "SIGRTMAX-8":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS8
			case "SIGRTMAX-7":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS7
			case "SIGRTMAX-6":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS6
			case "SIGRTMAX-5":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS5
			case "SIGRTMAX-4":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS4
			case "SIGRTMAX-3":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS3
			case "SIGRTMAX-2":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS2
			case "SIGRTMAX-1":
				signalValue = runtimeapi.Signal_SIGRTMAXMINUS1
			case "SIGRTMAX":
				signalValue = runtimeapi.Signal_SIGRTMAX
			}

			lifecycle := runtimeapi.Lifecycle{
				StopSignal: signalValue,
			}
			return &lifecycle
		} else {
			return nil
		}
	}

	return nil
}

func fromCRIStopSignal(signal runtimeapi.Signal) *v1.Signal {
	var signalStr v1.Signal
	switch signal {
	case runtimeapi.Signal_SIGABRT:
		signalStr = "SIGABRT"
	case runtimeapi.Signal_SIGALRM:
		signalStr = "SIGALRM"
	case runtimeapi.Signal_SIGBUS:
		signalStr = "SIGBUS"
	case runtimeapi.Signal_SIGCHLD:
		signalStr = "SIGCHLD"
	case runtimeapi.Signal_SIGCLD:
		signalStr = "SIGCLD"
	case runtimeapi.Signal_SIGCONT:
		signalStr = "SIGCONT"
	case runtimeapi.Signal_SIGFPE:
		signalStr = "SIGFPE"
	case runtimeapi.Signal_SIGHUP:
		signalStr = "SIGHUP"
	case runtimeapi.Signal_SIGILL:
		signalStr = "SIGILL"
	case runtimeapi.Signal_SIGINT:
		signalStr = "SIGINT"
	case runtimeapi.Signal_SIGIO:
		signalStr = "SIGIO"
	case runtimeapi.Signal_SIGIOT:
		signalStr = "SIGIOT"
	case runtimeapi.Signal_SIGKILL:
		signalStr = "SIGKILL"
	case runtimeapi.Signal_SIGPIPE:
		signalStr = "SIGPIPE"
	case runtimeapi.Signal_SIGPOLL:
		signalStr = "SIGPOLL"
	case runtimeapi.Signal_SIGPROF:
		signalStr = "SIGPROF"
	case runtimeapi.Signal_SIGPWR:
		signalStr = "SIGPWR"
	case runtimeapi.Signal_SIGQUIT:
		signalStr = "SIGQUIT"
	case runtimeapi.Signal_SIGSEGV:
		signalStr = "SIGSEGV"
	case runtimeapi.Signal_SIGSTKFLT:
		signalStr = "SIGSTKFLT"
	case runtimeapi.Signal_SIGSTOP:
		signalStr = "SIGSTOP"
	case runtimeapi.Signal_SIGSYS:
		signalStr = "SIGSYS"
	case runtimeapi.Signal_SIGTERM:
		signalStr = "SIGTERM"
	case runtimeapi.Signal_SIGTRAP:
		signalStr = "SIGTRAP"
	case runtimeapi.Signal_SIGTSTP:
		signalStr = "SIGTSTP"
	case runtimeapi.Signal_SIGTTIN:
		signalStr = "SIGTTIN"
	case runtimeapi.Signal_SIGTTOU:
		signalStr = "SIGTTOU"
	case runtimeapi.Signal_SIGURG:
		signalStr = "SIGURG"
	case runtimeapi.Signal_SIGUSR1:
		signalStr = "SIGUSR1"
	case runtimeapi.Signal_SIGUSR2:
		signalStr = "SIGUSR2"
	case runtimeapi.Signal_SIGVTALRM:
		signalStr = "SIGVTALRM"
	case runtimeapi.Signal_SIGWINCH:
		signalStr = "SIGWINCH"
	case runtimeapi.Signal_SIGXCPU:
		signalStr = "SIGXCPU"
	case runtimeapi.Signal_SIGXFSZ:
		signalStr = "SIGXFSZ"
	case runtimeapi.Signal_SIGRTMIN:
		signalStr = "SIGRTMIN"
	case runtimeapi.Signal_SIGRTMINPLUS1:
		signalStr = "SIGRTMIN+1"
	case runtimeapi.Signal_SIGRTMINPLUS2:
		signalStr = "SIGRTMIN+2"
	case runtimeapi.Signal_SIGRTMINPLUS3:
		signalStr = "SIGRTMIN+3"
	case runtimeapi.Signal_SIGRTMINPLUS4:
		signalStr = "SIGRTMIN+4"
	case runtimeapi.Signal_SIGRTMINPLUS5:
		signalStr = "SIGRTMIN+5"
	case runtimeapi.Signal_SIGRTMINPLUS6:
		signalStr = "SIGRTMIN+6"
	case runtimeapi.Signal_SIGRTMINPLUS7:
		signalStr = "SIGRTMIN+7"
	case runtimeapi.Signal_SIGRTMINPLUS8:
		signalStr = "SIGRTMIN+8"
	case runtimeapi.Signal_SIGRTMINPLUS9:
		signalStr = "SIGRTMIN+9"
	case runtimeapi.Signal_SIGRTMINPLUS10:
		signalStr = "SIGRTMIN+10"
	case runtimeapi.Signal_SIGRTMINPLUS11:
		signalStr = "SIGRTMIN+11"
	case runtimeapi.Signal_SIGRTMINPLUS12:
		signalStr = "SIGRTMIN+12"
	case runtimeapi.Signal_SIGRTMINPLUS13:
		signalStr = "SIGRTMIN+13"
	case runtimeapi.Signal_SIGRTMINPLUS14:
		signalStr = "SIGRTMIN+14"
	case runtimeapi.Signal_SIGRTMINPLUS15:
		signalStr = "SIGRTMIN+15"
	case runtimeapi.Signal_SIGRTMAXMINUS14:
		signalStr = "SIGRTMAX-14"
	case runtimeapi.Signal_SIGRTMAXMINUS13:
		signalStr = "SIGRTMAX-13"
	case runtimeapi.Signal_SIGRTMAXMINUS12:
		signalStr = "SIGRTMAX-12"
	case runtimeapi.Signal_SIGRTMAXMINUS11:
		signalStr = "SIGRTMAX-11"
	case runtimeapi.Signal_SIGRTMAXMINUS10:
		signalStr = "SIGRTMAX-10"
	case runtimeapi.Signal_SIGRTMAXMINUS9:
		signalStr = "SIGRTMAX-9"
	case runtimeapi.Signal_SIGRTMAXMINUS8:
		signalStr = "SIGRTMAX-8"
	case runtimeapi.Signal_SIGRTMAXMINUS7:
		signalStr = "SIGRTMAX-7"
	case runtimeapi.Signal_SIGRTMAXMINUS6:
		signalStr = "SIGRTMAX-6"
	case runtimeapi.Signal_SIGRTMAXMINUS5:
		signalStr = "SIGRTMAX-5"
	case runtimeapi.Signal_SIGRTMAXMINUS4:
		signalStr = "SIGRTMAX-4"
	case runtimeapi.Signal_SIGRTMAXMINUS3:
		signalStr = "SIGRTMAX-3"
	case runtimeapi.Signal_SIGRTMAXMINUS2:
		signalStr = "SIGRTMAX-2"
	case runtimeapi.Signal_SIGRTMAXMINUS1:
		signalStr = "SIGRTMAX-1"
	case runtimeapi.Signal_SIGRTMAX:
		signalStr = "SIGRTMAX"
	}
	return &signalStr
}
