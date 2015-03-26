/*
Copyright 2015 Google Inc. All rights reserved.

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
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

// Runtime interface defines the interfaces that should be implemented
// by a container runtime.
type Runtime interface {
	// Version returns a map of version information of the container runtime.
	Version() (map[string]string, error)
	// GetPods returns a list containers group by pods. The boolean parameter
	// specifies whether the runtime returns all containers including those already
	// exited and dead containers (used for garbage collection).
	GetPods(all bool) ([]*Pod, error)
	// RunPod starts all the containers of a pod within a namespace.
	RunPod(*api.Pod, map[string]volume.Volume) error
	// KillPod kills all the containers of a pod.
	KillPod(*api.Pod) error
	// RunContainerInPod starts a container within the same namespace of a pod.
	RunContainerInPod(api.Container, *api.Pod, map[string]volume.Volume) error
	// KillContainerInPod kills a container in the pod.
	KillContainerInPod(api.Container, *api.Pod) error
	// GetPodStatus retrieves the status of the pod, including the information of
	// all containers in the pod.
	GetPodStatus(*Pod) (api.PodStatus, error)
	// GetContainers returns all containers on the node, including those are
	// not managed by kubelet. If 'all' is false, then only running containers
	// are returned.
	GetContainers(all bool) ([]*Container, error)
	// TODO(yifan): Pull/Remove images
}

// Pod is a group of containers, with the status of the pod.
type Pod struct {
	// The ID of the pod, which can be used to retrieve a particular pod
	// from the pod list returned by GetPods().
	ID types.UID
	// The name and namespace of the pod, which is readable by human.
	Name      string
	Namespace string
	// List of containers that belongs to this pod. It may contain only
	// running containers, or mixed with dead ones (when GetPods(true)).
	Containers []*Container
	// The status of the pod.
	// TODO(yifan): Inspect and get the statuses for all pods can be expensive,
	// maybe we want to get one pod's status at a time (e.g. GetPodStatus()
	// for the particular pod after we GetPods()).
	Status api.PodStatus
}

// Container provides the runtime information for a container, such as ID, hash,
// status of the container.
type Container struct {
	// The ID of the container, used by the container runtime to identify
	// a container.
	ID types.UID
	// The name of the container, which should be the same as specified by
	// api.Container.
	Name string
	// The image name of the container.
	Image string
	// Hash of the container, used for comparison. Optional for containers
	// not managed by kubelet.
	Hash uint64
	// The timestamp of the creation time of the container.
	// TODO(yifan): Consider to move it to api.ContainerStatus.
	Created int64
}

// RunContainerOptions specify the options which are necessary for running containers
type RunContainerOptions struct {
	// The environment variables, they are in the form of 'key=value'.
	Envs []string
	// The mounts for the containers, they are in the form of:
	// 'hostPath:containerPath', or
	// 'hostPath:containerPath:ro', if the path read only.
	Binds []string
	// If the container has specified the TerminationMessagePath, then
	// this directory will be used to create and mount the log file to
	// container.TerminationMessagePath
	PodContainerDir string
	// The list of DNS servers for the container to use.
	DNS []string
	// The list of DNS search domains.
	DNSSearch []string
	// Docker namespace identifiers(currently we have 'NetMode' and 'IpcMode'.
	// These are for docker to attach a container in a pod to the pod infra
	// container's namespace.
	// TODO(yifan): Remove these after we pushed the pod infra container logic
	// into docker's container runtime.
	NetMode string
	IpcMode string
}

type Pods []*Pod

// FindPodByID returns a pod in the pod list by UID. It will return an empty pod
// if not found.
// TODO(yifan): Use a map?
func (p Pods) FindPodByID(podUID types.UID) Pod {
	for i := range p {
		if p[i].ID == podUID {
			return *p[i]
		}
	}
	return Pod{}
}

// FindContainerByName returns a container in the pod with the given name.
// When there are multiple containers with the same name, the first match will
// be returned.
func (p *Pod) FindContainerByName(containerName string) *Container {
	for _, c := range p.Containers {
		if c.Name == containerName {
			return c
		}
	}
	return nil
}

// GetPodFullName returns a name that uniquely identifies a pod.
func GetPodFullName(pod *api.Pod) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format), while allowed in the container name format.
	return fmt.Sprintf("%s_%s", pod.Name, pod.Namespace)
}

// Build the pod full name from pod name and namespace.
func BuildPodFullName(name, namespace string) string {
	return name + "_" + namespace
}

// Parse the pod full name.
func ParsePodFullName(podFullName string) (string, string, error) {
	parts := strings.Split(podFullName, "_")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("failed to parse the pod full name %q", podFullName)
	}
	return parts[0], parts[1], nil
}
