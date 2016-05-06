/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package lifecycle

import "k8s.io/kubernetes/pkg/api"

// PodAdmitAttributes is the context for a pod admission decision.
// The member fields of this struct should never be mutated.
type PodAdmitAttributes struct {
	// the pod to evaluate for admission
	Pod *api.Pod
	// all pods bound to the kubelet excluding the pod being evaluated
	OtherPods []*api.Pod
}

// PodAdmitResult provides the result of a pod admission decision.
type PodAdmitResult struct {
	// if true, the pod should be admitted.
	Admit bool
	// a brief single-word reason why the pod could not be admitted.
	Reason string
	// a brief message explaining why the pod could not be admitted.
	Message string
}

// PodAdmitHandler is notified during pod admission.
type PodAdmitHandler interface {
	// Admit evaluates if a pod can be admitted.
	Admit(attrs *PodAdmitAttributes) PodAdmitResult
}

// PodAdmitTarget maintains a list of handlers to invoke.
type PodAdmitTarget interface {
	// AddPodAdmitHandler adds the specified handler.
	AddPodAdmitHandler(a PodAdmitHandler)
}

// PodSyncLoopHandler is invoked during each sync loop iteration.
type PodSyncLoopHandler interface {
	// ShouldSync returns true if the pod needs to be synced.
	// This operation must return immediately as its called for each pod.
	// The provided pod should never be modified.
	ShouldSync(pod *api.Pod) bool
}

// PodSyncLoopTarget maintains a list of handlers to pod sync loop.
type PodSyncLoopTarget interface {
	// AddPodSyncLoopHandler adds the specified handler.
	AddPodSyncLoopHandler(a PodSyncLoopHandler)
}

// ShouldEvictResponse provides the result of a should evict request.
type ShouldEvictResponse struct {
	// if true, the pod should be evicted.
	Evict bool
	// a brief CamelCase reason why the pod should be evicted.
	Reason string
	// a brief message why the pod should be evicted.
	Message string
}

// PodSyncHandler is invoked during each sync pod operation.
type PodSyncHandler interface {
	// ShouldEvict is invoked during each sync pod operation to determine
	// if the pod should be evicted from the kubelet.  If so, the pod status
	// is updated to mark its phase as failed with the provided reason and message,
	// and the pod is immediately killed.
	// This operation must return immediately as its called for each sync pod.
	// The provided pod should never be modified.
	ShouldEvict(pod *api.Pod) ShouldEvictResponse
}

// PodSyncTarget maintains a list of handlers to pod sync.
type PodSyncTarget interface {
	// AddPodSyncHandler adds the specified handler
	AddPodSyncHandler(a PodSyncHandler)
}

// PodLifecycleTarget groups a set of lifecycle interfaces for convenience.
type PodLifecycleTarget interface {
	PodAdmitTarget
	PodSyncLoopTarget
	PodSyncTarget
}

// PodAdmitHandlers maintains a list of handlers to pod admission.
type PodAdmitHandlers []PodAdmitHandler

// AddPodAdmitHandler adds the specified observer.
func (handlers *PodAdmitHandlers) AddPodAdmitHandler(a PodAdmitHandler) {
	*handlers = append(*handlers, a)
}

// PodSyncLoopHandlers maintains a list of handlers to pod sync loop.
type PodSyncLoopHandlers []PodSyncLoopHandler

// AddPodSyncLoopHandler adds the specified observer.
func (handlers *PodSyncLoopHandlers) AddPodSyncLoopHandler(a PodSyncLoopHandler) {
	*handlers = append(*handlers, a)
}

// PodSyncHandlers maintains a list of handlers to pod sync.
type PodSyncHandlers []PodSyncHandler

// AddPodSyncHandler adds the specified handler.
func (handlers *PodSyncHandlers) AddPodSyncHandler(a PodSyncHandler) {
	*handlers = append(*handlers, a)
}
