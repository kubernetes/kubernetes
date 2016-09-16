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
	"hash/adler32"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/third_party/forked/golang/expansion"

	"github.com/golang/glog"
)

// HandlerRunner runs a lifecycle handler for a container.
type HandlerRunner interface {
	Run(containerID ContainerID, pod *api.Pod, container *api.Container, handler *api.Handler) (string, error)
}

// RuntimeHelper wraps kubelet to make container runtime
// able to get necessary informations like the RunContainerOptions, DNS settings.
type RuntimeHelper interface {
	GenerateRunContainerOptions(pod *api.Pod, container *api.Container, podIP string) (*RunContainerOptions, error)
	GetClusterDNS(pod *api.Pod) (dnsServers []string, dnsSearches []string, err error)
	GetPodDir(podUID types.UID) string
	GeneratePodHostNameAndDomain(pod *api.Pod) (hostname string, hostDomain string, err error)
	// GetExtraSupplementalGroupsForPod returns a list of the extra
	// supplemental groups for the Pod. These extra supplemental groups come
	// from annotations on persistent volumes that the pod depends on.
	GetExtraSupplementalGroupsForPod(pod *api.Pod) []int64
}

// ShouldContainerBeRestarted checks whether a container needs to be restarted.
// TODO(yifan): Think about how to refactor this.
func ShouldContainerBeRestarted(container *api.Container, pod *api.Pod, podStatus *PodStatus) bool {
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
	if pod.Spec.RestartPolicy == api.RestartPolicyNever {
		glog.V(4).Infof("Already ran container %q of pod %q, do nothing", container.Name, format.Pod(pod))
		return false
	}
	if pod.Spec.RestartPolicy == api.RestartPolicyOnFailure {
		// Check the exit code.
		if status.ExitCode == 0 {
			glog.V(4).Infof("Already successfully ran container %q of pod %q, do nothing", container.Name, format.Pod(pod))
			return false
		}
	}
	return true
}

// TODO(random-liu): Convert PodStatus to running Pod, should be deprecated soon
func ConvertPodStatusToRunningPod(podStatus *PodStatus) Pod {
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
	return runningPod
}

// HashContainer returns the hash of the container. It is used to compare
// the running container with its desired spec.
func HashContainer(container *api.Container) uint64 {
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

func ExpandContainerCommandAndArgs(container *api.Container, envs []EnvVar) (command []string, args []string) {
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

func (irecorder *innerEventRecorder) shouldRecordEvent(object runtime.Object) (*api.ObjectReference, bool) {
	if object == nil {
		return nil, false
	}
	if ref, ok := object.(*api.ObjectReference); ok {
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

func (irecorder *innerEventRecorder) PastEventf(object runtime.Object, timestamp unversioned.Time, eventtype, reason, messageFmt string, args ...interface{}) {
	if ref, ok := irecorder.shouldRecordEvent(object); ok {
		irecorder.recorder.PastEventf(ref, timestamp, eventtype, reason, messageFmt, args...)
	}
}

// Pod must not be nil.
func IsHostNetworkPod(pod *api.Pod) bool {
	return pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork
}
