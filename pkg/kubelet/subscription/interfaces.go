/*
Copyright 2025 The Kubernetes Authors.

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

package subscription

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

// PodUpdate is a union of a pod update and a pod status update.
type PodUpdate struct {
	Pod    *v1.Pod
	Update types.SyncPodType
}

// PodUpdateSubscriber is an interface for other Kubelet components to subscribe to pod updates.
type PodUpdateSubscriber interface {
	// OnPodAdded is called when a pod is added.
	OnPodAdded(pod *v1.Pod)
	// OnPodUpdated is called when a pod is updated.
	OnPodUpdated(pod *v1.Pod)
	// OnPodRemoved is called when a pod is removed.
	OnPodRemoved(pod *v1.Pod)
}

// StatusUpdate is a union of a pod and a pod status.
type StatusUpdate struct {
	Pod    *v1.Pod
	Status v1.PodStatus
}

// StatusUpdateSubscriber is an interface for other Kubelet components to subscribe to pod status updates.
type StatusUpdateSubscriber interface {
	// OnPodStatusUpdated is called when a pod's status is updated.
	OnPodStatusUpdated(pod *v1.Pod, status v1.PodStatus)
}
