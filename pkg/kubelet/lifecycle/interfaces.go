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

package lifecycle

import (
	"context"

	v1 "k8s.io/api/core/v1"
)

// PodAdmitAttributes is the context for a pod admission decision.
// Top-level fields describe the admission itself, while Update holds facts about the
// allocated -> desired change precomputed once by the allocation manager.
// The member fields of this struct should never be mutated.
type PodAdmitAttributes struct {
	// the pod to evaluate for admission
	Pod *v1.Pod
	// all pods bound to the kubelet excluding the pod being evaluated
	OtherPods []*v1.Pod
	// the operation being performed; either "add" or "update"
	Operation Operation
	// Update holds information about the pod update being evaluated.
	// It is set if and only if Operation == UpdateOperation (nil for AddOperation).
	Update *PodUpdate
	// DryRun indicates that the pod won't be admitted regardless of the result, and that the
	// handler must not cause side effects. See PodAdmitHandler.
	DryRun bool
}

// PodUpdate holds facts about the allocated -> desired change for an UpdateOperation,
// precomputed once by the allocation manager.
type PodUpdate struct {
	// AllocatedPod is the pod as it is currently allocated on the node.
	AllocatedPod *v1.Pod
	// Resize is true if the update changes resources.
	Resize bool
}

// Operation represents the type of operation being performed on a pod.
type Operation string

const (
	// AddOperation indicates that the pod is being added to the kubelet.
	AddOperation Operation = "add"
	// UpdateOperation indicates that an existing pod's allocation is being updated.
	UpdateOperation Operation = "update"
)

// AdmitResult is the final outcome of an admission, given to
// PodAdmitTransactionalHandler.CompleteAdmit.
type AdmitResult string

const (
	// AdmitSuccess indicates that every handler admitted the pod and the pod
	// allocation has been persisted.
	AdmitSuccess AdmitResult = "success"
	// AdmitFailure indicates that the pod was rejected, or that the admission
	// failed before the pod allocation was persisted.
	AdmitFailure AdmitResult = "failure"
)

// PodAdmitResult provides the result of a pod admission decision.
type PodAdmitResult struct {
	// if true, the pod should be admitted.
	Admit bool
	// a brief single-word reason why the pod could not be admitted.
	Reason string
	// a brief message explaining why the pod could not be admitted.
	Message string
	// RejectionType classifies why the pod could not be admitted. Only meaningful if Admit is false.
	RejectionType RejectionType
}

// RejectionType classifies a pod admission rejection by what can resolve it.
// Rejection types are ordered by increasing precedence: when several handlers reject a pod, the rejection
// with the highest rejection type is reported.
type RejectionType int

const (
	// RejectionTransient means that retrying without any other change may succeed, for example
	// after an internal error. This is the zero value, so an unclassified rejection is retried.
	RejectionTransient RejectionType = iota
	// RejectionResourceShortage means that the node doesn't have enough of a resource that the
	// scheduler accounts for. Preempting, resizing or terminating other pods can resolve it.
	RejectionResourceShortage
	// RejectionBlocked means that the rejection can only be resolved by some other change
	// in the state of the node or its pods, such as freeing NUMA-aligned resources or relieving
	// node pressure. Retrying without that change won't help, and preemption based on aggregate
	// resource requests is not expected to help either.
	RejectionBlocked
	// RejectionInfeasible means that the pod can't be admitted without changing the pod spec or
	// the node configuration.
	RejectionInfeasible
)

// PodAdmitHandler is notified during pod admission.
type PodAdmitHandler interface {
	// Admit evaluates if a pod can be admitted.
	// If attrs.DryRun is set, Admit must not cause side effects, such as evicting or preempting
	// pods, allocating resources, or making external calls that change state, but it should
	// still report whether it would reject the pod. This mirrors dry run in API admission.
	Admit(ctx context.Context, attrs *PodAdmitAttributes) PodAdmitResult
}

// PodAdmitTransactionalHandler is an optional extension of PodAdmitHandler for handlers
// that prepare a resource assignment in Admit. Such an assignment must not
// become effective before every handler has admitted the pod and the pod
// allocation has been persisted, because a rejected update, such as a resize,
// leaves the pod running and nothing else would clean the assignment up.
//
// Admit prepares the assignment without making it effective.
// Once the outcome of the admission is final, the caller invokes CompleteAdmit
// exactly once: with AdmitSuccess after the pod allocation has been persisted,
// and with AdmitFailure on a rejection or any failure before that.
// The caller invokes CompleteAdmit on every handler that implements this
// interface, also on a handler whose Admit was not called, or was called with
// attrs.DryRun set, because an earlier handler rejected the pod. Such an
// admission always ends with AdmitFailure, so whatever Admit prepared in dry
// run, which must never be effective, is discarded.
type PodAdmitTransactionalHandler interface {
	// CompleteAdmit makes the assignment prepared in Admit effective on
	// AdmitSuccess, and discards it on AdmitFailure. It cannot fail: Admit
	// checks everything that could prevent the commit, and the persisted pod
	// allocation is not rolled back, so an inconsistency found here is an
	// implementation bug, which the handler logs. On AdmitFailure, it must do
	// nothing if Admit was not called or did not prepare anything for the pod.
	CompleteAdmit(ctx context.Context, attrs *PodAdmitAttributes, result AdmitResult)
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
	ShouldSync(pod *v1.Pod) bool
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
	ShouldEvict(pod *v1.Pod) ShouldEvictResponse
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
