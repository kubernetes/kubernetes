/*
Copyright The Kubernetes Authors.

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

package pod

import (
	"fmt"
	"maps"
	"slices"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/kubernetes/pkg/apis/core"
)

// PodAccessor provides mutation-safe read access (and metav1.Object metadata
// updates) to core.Pod fields without requiring the caller to assert a concrete
// *core.Pod struct.
//
// Getters return either immutable values or deep copies so that mutating a
// returned slice, map, or pointer cannot modify the underlying object, and
// direct mutations to returned values are never reflected back into the object.
// Callers updating metadata must pass modified copies via metav1.Object setters
// (such as SetLabels or SetAnnotations).
//
// +k8s:deepcopy-gen=false
type PodAccessor interface {
	runtime.Object
	metav1.Object
	GetSpecInitContainers() []core.Container
	GetSpecContainers() []core.Container
	GetSpecEphemeralContainers() []core.EphemeralContainer
	GetSpecRestartPolicy() core.RestartPolicy
	GetSpecTerminationGracePeriodSeconds() (int64, bool)
	GetSpecServiceAccountName() string
	GetSpecNodeName() string
	GetSpecHostNetwork() bool
	GetSpecAffinity() *core.Affinity
	GetSpecSchedulerName() string
	GetStatusPhase() core.PodPhase
	GetStatusNominatedNodeName() string
	GetStatusHostIPs() []core.HostIP
	GetStatusPodIPs() []core.PodIP
	GetStatusContainerStatuses() []core.ContainerStatus
	GetStatusResourceClaimStatuses() []core.PodResourceClaimStatus
	GetStatusExtendedResourceClaimStatus() *core.PodExtendedResourceClaimStatus
}

// Accessor returns a PodAccessor for the provided object, or an error if the object
// does not implement PodAccessor or *core.Pod.
func Accessor(obj any) (PodAccessor, error) {
	switch t := obj.(type) {
	case PodAccessor:
		return t, nil
	case *core.Pod:
		if t == nil {
			return nil, nil
		}
		return &podAccessor{Pod: t}, nil
	default:
		return nil, fmt.Errorf("object %T does not implement PodAccessor", obj)
	}
}

// MustAccessor returns a PodAccessor for the provided object, or panics if the object
// does not implement PodAccessor or *core.Pod.
func MustAccessor(obj any) PodAccessor {
	acc, err := Accessor(obj)
	if err != nil {
		panic(err)
	}
	return acc
}

type podAccessor struct {
	*core.Pod
}

var _ PodAccessor = (*podAccessor)(nil)

func (in *podAccessor) DeepCopyObject() runtime.Object {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.DeepCopyObject()
}

func (in *podAccessor) GetDeletionTimestamp() *metav1.Time {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.DeletionTimestamp.DeepCopy()
}

func (in *podAccessor) SetDeletionTimestamp(timestamp *metav1.Time) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.DeletionTimestamp = timestamp.DeepCopy()
}

func (in *podAccessor) GetDeletionGracePeriodSeconds() *int64 {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneInt64Ptr(in.Pod.DeletionGracePeriodSeconds)
}

func (in *podAccessor) SetDeletionGracePeriodSeconds(gracePeriodSeconds *int64) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.DeletionGracePeriodSeconds = cloneInt64Ptr(gracePeriodSeconds)
}

func (in *podAccessor) GetLabels() map[string]string {
	if in == nil || in.Pod == nil {
		return nil
	}
	return maps.Clone(in.Pod.Labels)
}

func (in *podAccessor) SetLabels(labels map[string]string) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.Labels = maps.Clone(labels)
}

func (in *podAccessor) GetAnnotations() map[string]string {
	if in == nil || in.Pod == nil {
		return nil
	}
	return maps.Clone(in.Pod.Annotations)
}

func (in *podAccessor) SetAnnotations(annotations map[string]string) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.Annotations = maps.Clone(annotations)
}

func (in *podAccessor) GetFinalizers() []string {
	if in == nil || in.Pod == nil {
		return nil
	}
	return slices.Clone(in.Pod.Finalizers)
}

func (in *podAccessor) SetFinalizers(finalizers []string) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.Finalizers = slices.Clone(finalizers)
}

func (in *podAccessor) GetOwnerReferences() []metav1.OwnerReference {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneOwnerReferences(in.Pod.OwnerReferences)
}

func (in *podAccessor) SetOwnerReferences(references []metav1.OwnerReference) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.OwnerReferences = cloneOwnerReferences(references)
}

func (in *podAccessor) GetManagedFields() []metav1.ManagedFieldsEntry {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneManagedFields(in.Pod.ManagedFields)
}

func (in *podAccessor) SetManagedFields(managedFields []metav1.ManagedFieldsEntry) {
	if in == nil || in.Pod == nil {
		return
	}
	in.Pod.ManagedFields = cloneManagedFields(managedFields)
}

func (in *podAccessor) GetSpecInitContainers() []core.Container {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneContainers(in.Pod.Spec.InitContainers)
}

func (in *podAccessor) GetSpecContainers() []core.Container {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneContainers(in.Pod.Spec.Containers)
}

func (in *podAccessor) GetSpecEphemeralContainers() []core.EphemeralContainer {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneEphemeralContainers(in.Pod.Spec.EphemeralContainers)
}

func (in *podAccessor) GetSpecRestartPolicy() core.RestartPolicy {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.RestartPolicy
}

func (in *podAccessor) GetSpecTerminationGracePeriodSeconds() (int64, bool) {
	if in == nil || in.Pod == nil || in.Pod.Spec.TerminationGracePeriodSeconds == nil {
		return 0, false
	}
	return *in.Pod.Spec.TerminationGracePeriodSeconds, true
}

func (in *podAccessor) GetSpecServiceAccountName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.ServiceAccountName
}

func (in *podAccessor) GetSpecNodeName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.NodeName
}

func (in *podAccessor) GetSpecHostNetwork() bool {
	if in == nil || in.Pod == nil {
		return false
	}
	return in.Pod.Spec.HostNetwork
}

func (in *podAccessor) GetSpecAffinity() *core.Affinity {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.Affinity.DeepCopy()
}

func (in *podAccessor) GetSpecSchedulerName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.SchedulerName
}

func (in *podAccessor) GetStatusPhase() core.PodPhase {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Status.Phase
}

func (in *podAccessor) GetStatusNominatedNodeName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Status.NominatedNodeName
}

func (in *podAccessor) GetStatusHostIPs() []core.HostIP {
	if in == nil || in.Pod == nil {
		return nil
	}
	return slices.Clone(in.Pod.Status.HostIPs)
}

func (in *podAccessor) GetStatusPodIPs() []core.PodIP {
	if in == nil || in.Pod == nil {
		return nil
	}
	return slices.Clone(in.Pod.Status.PodIPs)
}

func (in *podAccessor) GetStatusContainerStatuses() []core.ContainerStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneContainerStatuses(in.Pod.Status.ContainerStatuses)
}

func (in *podAccessor) GetStatusResourceClaimStatuses() []core.PodResourceClaimStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return cloneResourceClaimStatuses(in.Pod.Status.ResourceClaimStatuses)
}

func (in *podAccessor) GetStatusExtendedResourceClaimStatus() *core.PodExtendedResourceClaimStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.ExtendedResourceClaimStatus.DeepCopy()
}

func cloneInt64Ptr(v *int64) *int64 {
	if v == nil {
		return nil
	}
	copied := *v
	return &copied
}

func cloneContainers(src []core.Container) []core.Container {
	if src == nil {
		return nil
	}
	out := make([]core.Container, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}

func cloneEphemeralContainers(src []core.EphemeralContainer) []core.EphemeralContainer {
	if src == nil {
		return nil
	}
	out := make([]core.EphemeralContainer, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}

func cloneContainerStatuses(src []core.ContainerStatus) []core.ContainerStatus {
	if src == nil {
		return nil
	}
	out := make([]core.ContainerStatus, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}

func cloneResourceClaimStatuses(src []core.PodResourceClaimStatus) []core.PodResourceClaimStatus {
	if src == nil {
		return nil
	}
	out := make([]core.PodResourceClaimStatus, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}

func cloneOwnerReferences(src []metav1.OwnerReference) []metav1.OwnerReference {
	if src == nil {
		return nil
	}
	out := make([]metav1.OwnerReference, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}

func cloneManagedFields(src []metav1.ManagedFieldsEntry) []metav1.ManagedFieldsEntry {
	if src == nil {
		return nil
	}
	out := make([]metav1.ManagedFieldsEntry, len(src))
	for i := range src {
		src[i].DeepCopyInto(&out[i])
	}
	return out
}
