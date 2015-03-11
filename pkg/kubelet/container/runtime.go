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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
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
	RunPod(*api.Pod, map[string]volume.Interface) error
	// KillPod kills all the containers of a pod.
	KillPod(*api.Pod) error
	// RunContainerInPod starts a container within the same namespace of a pod.
	RunContainerInPod(api.Container, *api.Pod, map[string]volume.Interface) error
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
}
