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

package helper

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha1"
	"k8s.io/klog/v2"
)

// MatchingWorkloadReference returns true if two pods belong to the same workload, including their pod group and replica key.
func MatchingWorkloadReference(pod1, pod2 *v1.Pod) bool {
	return pod1.Spec.WorkloadRef != nil && pod2.Spec.WorkloadRef != nil && pod1.Namespace == pod2.Namespace && *pod1.Spec.WorkloadRef == *pod2.Spec.WorkloadRef
}

// PodGroupPolicy is a helper to find the policy for a specific pod group name in a workload.
func PodGroupPolicy(workload *schedulingapi.Workload, podGroupName string) (schedulingapi.PodGroupPolicy, bool) {
	for _, podGroup := range workload.Spec.PodGroups {
		if podGroup.Name == podGroupName {
			return podGroup.Policy, true
		}
	}
	return schedulingapi.PodGroupPolicy{}, false
}

// HasDisruptionModePodGroup checks if the pod is a part of a gang pod group
// with the PodGroup disruption mode.
func HasDisruptionModePodGroup(p *v1.Pod, wl schedulinglisters.WorkloadLister) bool {
	if wl == nil || p.Spec.WorkloadRef == nil {
		return false
	}
	workload, err := wl.Workloads(p.Namespace).Get(p.Spec.WorkloadRef.Name)
	if err != nil {
		return false
	}
	policy, ok := PodGroupPolicy(workload, p.Spec.WorkloadRef.PodGroup)
	if !ok || policy.Gang == nil || policy.Gang.DisruptionMode == nil {
		return false
	}
	return *policy.Gang.DisruptionMode == schedulingapi.DisruptionModePodGroup
}

// PodGroupKey uniquely identifies a specific instance of a PodGroup.
type PodGroupKey struct {
	namespace    string
	workloadName string
	podGroupName string
	replicaKey   string
}

var _ klog.KMetadata = &PodGroupKey{}

func (pgk *PodGroupKey) GetName() string {
	if pgk.replicaKey == "" {
		return fmt.Sprintf("%s-%s", pgk.workloadName, pgk.podGroupName)
	}
	return fmt.Sprintf("%s-%s-%s", pgk.workloadName, pgk.podGroupName, pgk.replicaKey)
}

func (pgk *PodGroupKey) GetNamespace() string {
	return pgk.namespace
}

func (pgk *PodGroupKey) GetWorkloadRef() *v1.WorkloadReference {
	return &v1.WorkloadReference{
		Name:               pgk.workloadName,
		PodGroup:           pgk.podGroupName,
		PodGroupReplicaKey: pgk.replicaKey,
	}
}

func NewPodGroupKey(namespace string, workloadRef *v1.WorkloadReference) PodGroupKey {
	return PodGroupKey{
		namespace:    namespace,
		workloadName: workloadRef.Name,
		podGroupName: workloadRef.PodGroup,
		replicaKey:   workloadRef.PodGroupReplicaKey,
	}
}

// GetPodGroupKey returns the key of the pod group that the pod belongs to.
// If the pod does not belong to any workload, it returns nil.
func GetPodGroupKey(p *v1.Pod) *PodGroupKey {
	if p.Spec.WorkloadRef == nil {
		return nil
	}
	return &PodGroupKey{
		namespace:    p.Namespace,
		workloadName: p.Spec.WorkloadRef.Name,
		podGroupName: p.Spec.WorkloadRef.PodGroup,
		replicaKey:   p.Spec.WorkloadRef.PodGroupReplicaKey,
	}
}
