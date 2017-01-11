/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"hash/adler32"
	"strings"
	"time"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/third_party/forked/golang/expansion"
)

// HandlerRunner runs a lifecycle handler for a container.
type HandlerRunner interface {
	Run(containerID ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.Handler) (string, error)
}

// RuntimeHelper wraps kubelet to make container runtime
// able to get necessary informations like the RunContainerOptions, DNS settings.
type RuntimeHelper interface {
	GenerateRunContainerOptions(pod *v1.Pod, container *v1.Container, podIP string) (*RunContainerOptions, error)
	GetClusterDNS(pod *v1.Pod) (dnsServers []string, dnsSearches []string, err error)
	GetPodDir(podUID types.UID) string
	GeneratePodHostNameAndDomain(pod *v1.Pod) (hostname string, hostDomain string, err error)
	// GetExtraSupplementalGroupsForPod returns a list of the extra
	// supplemental groups for the Pod. These extra supplemental groups come
	// from annotations on persistent volumes that the pod depends on.
	GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64
}

// ShouldContainerBeRestarted checks whether a container needs to be restarted.
// TODO(yifan): Think about how to refactor this.
func ShouldContainerBeRestarted(container *v1.Container, pod *v1.Pod, podStatus *PodStatus) bool {
	// Get latest container status.
	status := podStatus.FindContainerStatusByName(container.Name)
	// If the container was never started before, we should start it.
	// NOTE(random-liu): If all historical containers were GC'd, we'll also return true here.
	if status == nil {
		return true
	}
	// Check whether container is running
	if status.State == ContainerStateRunning {
		return false
	}
	// Always restart container in unknown state now
	if status.State == ContainerStateUnknown {
		return true
	}
	// Check RestartPolicy for dead container
	if pod.Spec.RestartPolicy == v1.RestartPolicyNever {
		glog.V(4).Infof("Already ran container %q of pod %q, do nothing", container.Name, format.Pod(pod))
		return false
	}
	if pod.Spec.RestartPolicy == v1.RestartPolicyOnFailure {
		// Check the exit code.
		if status.ExitCode == 0 {
			glog.V(4).Infof("Already successfully ran container %q of pod %q, do nothing", container.Name, format.Pod(pod))
			return false
		}
	}
	return true
}

// HashContainer returns the hash of the container. It is used to compare
// the running container with its desired spec.
func HashContainer(container *v1.Container) uint64 {
	hash := adler32.New()
	hashutil.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}

// EnvVarsToMap constructs a map of environment name to value from a slice
// of env vars.
func EnvVarsToMap(envs []EnvVar) map[string]string {
	result := map[string]string{}
	for _, env := range envs {
		result[env.Name] = env.Value
	}

	return result
}

func ExpandContainerCommandAndArgs(container *v1.Container, envs []EnvVar) (command []string, args []string) {
	mapping := expansion.MappingFuncFor(EnvVarsToMap(envs))

	if len(container.Command) != 0 {
		for _, cmd := range container.Command {
			command = append(command, expansion.Expand(cmd, mapping))
		}
	}

	if len(container.Args) != 0 {
		for _, arg := range container.Args {
			args = append(args, expansion.Expand(arg, mapping))
		}
	}

	return command, args
}

// Create an event recorder to record object's event except implicitly required container's, like infra container.
func FilterEventRecorder(recorder record.EventRecorder) record.EventRecorder {
	return &innerEventRecorder{
		recorder: recorder,
	}
}

type innerEventRecorder struct {
	recorder record.EventRecorder
}

func (irecorder *innerEventRecorder) shouldRecordEvent(object runtime.Object) (*v1.ObjectReference, bool) {
	if object == nil {
		return nil, false
	}
	if ref, ok := object.(*v1.ObjectReference); ok {
		if !strings.HasPrefix(ref.FieldPath, ImplicitContainerPrefix) {
			return ref, true
		}
	}
	return nil, false
}

func (irecorder *innerEventRecorder) Event(object runtime.Object, eventtype, reason, message string) {
	if ref, ok := irecorder.shouldRecordEvent(object); ok {
		irecorder.recorder.Event(ref, eventtype, reason, message)
	}
}

func (irecorder *innerEventRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	if ref, ok := irecorder.shouldRecordEvent(object); ok {
		irecorder.recorder.Eventf(ref, eventtype, reason, messageFmt, args...)
	}

}

func (irecorder *innerEventRecorder) PastEventf(object runtime.Object, timestamp metav1.Time, eventtype, reason, messageFmt string, args ...interface{}) {
	if ref, ok := irecorder.shouldRecordEvent(object); ok {
		irecorder.recorder.PastEventf(ref, timestamp, eventtype, reason, messageFmt, args...)
	}
}

// Pod must not be nil.
func IsHostNetworkPod(pod *v1.Pod) bool {
	return pod.Spec.HostNetwork
}

// TODO(random-liu): Convert PodStatus to running Pod, should be deprecated soon
func ConvertPodStatusToRunningPod(runtimeName string, podStatus *PodStatus) Pod {
	runningPod := Pod{
		ID:        podStatus.ID,
		Name:      podStatus.Name,
		Namespace: podStatus.Namespace,
	}
	for _, containerStatus := range podStatus.ContainerStatuses {
		if containerStatus.State != ContainerStateRunning {
			continue
		}
		container := &Container{
			ID:      containerStatus.ID,
			Name:    containerStatus.Name,
			Image:   containerStatus.Image,
			ImageID: containerStatus.ImageID,
			Hash:    containerStatus.Hash,
			State:   containerStatus.State,
		}
		runningPod.Containers = append(runningPod.Containers, container)
	}

	// Populate sandboxes in kubecontainer.Pod
	for _, sandbox := range podStatus.SandboxStatuses {
		runningPod.Sandboxes = append(runningPod.Sandboxes, &Container{
			ID:    ContainerID{Type: runtimeName, ID: *sandbox.Id},
			State: SandboxToContainerState(*sandbox.State),
		})
	}
	return runningPod
}

// sandboxToContainerState converts runtimeApi.PodSandboxState to
// kubecontainer.ContainerState.
// This is only needed because we need to return sandboxes as if they were
// kubecontainer.Containers to avoid substantial changes to PLEG.
// TODO: Remove this once it becomes obsolete.
func SandboxToContainerState(state runtimeapi.PodSandboxState) ContainerState {
	switch state {
	case runtimeapi.PodSandboxState_SANDBOX_READY:
		return ContainerStateRunning
	case runtimeapi.PodSandboxState_SANDBOX_NOTREADY:
		return ContainerStateExited
	}
	return ContainerStateUnknown
}

// FormatPod returns a string representing a pod in a human readable format,
// with pod UID as part of the string.
func FormatPod(pod *Pod) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format), while allowed in the container name format.
	return fmt.Sprintf("%s_%s(%s)", pod.Name, pod.Namespace, pod.ID)
}

type containerCommandRunnerWrapper struct {
	DirectStreamingRuntime
}

var _ ContainerCommandRunner = &containerCommandRunnerWrapper{}

func DirectStreamingRunner(runtime DirectStreamingRuntime) ContainerCommandRunner {
	return &containerCommandRunnerWrapper{runtime}
}

func (r *containerCommandRunnerWrapper) RunInContainer(id ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	var buffer bytes.Buffer
	output := ioutils.WriteCloserWrapper(&buffer)
	err := r.ExecInContainer(id, cmd, nil, output, output, false, nil, timeout)
	// Even if err is non-nil, there still may be output (e.g. the exec wrote to stdout or stderr but
	// the command returned a nonzero exit code). Therefore, always return the output along with the
	// error.
	return buffer.Bytes(), err
}

// GetContainerSpec gets the container spec by containerName.
func GetContainerSpec(pod *v1.Pod, containerName string) *v1.Container {
	for i, c := range pod.Spec.Containers {
		if containerName == c.Name {
			return &pod.Spec.Containers[i]
		}
	}
	for i, c := range pod.Spec.InitContainers {
		if containerName == c.Name {
			return &pod.Spec.InitContainers[i]
		}
	}
	return nil
}

// HasPrivilegedContainer returns true if any of the containers in the pod are privileged.
func HasPrivilegedContainer(pod *v1.Pod) bool {
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext != nil &&
			c.SecurityContext.Privileged != nil &&
			*c.SecurityContext.Privileged {
			return true
		}
	}
	return false
}
