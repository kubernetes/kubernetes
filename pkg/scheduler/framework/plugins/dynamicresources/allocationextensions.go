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

package dynamicresources

import (
	"context"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// This file implements framework.AllocationExtensions (KEP-5690). For now,
// the plugin only needs the node which the PodNominator nominated a pod to:
// the nominated node in the pod object lags behind while asynchronous API
// calls are pending.

var _ framework.AllocationExtensions = &DynamicResources{}

// PrepareAllocation does nothing because the plugin keeps no allocations
// yet.
func (pl *DynamicResources) PrepareAllocation(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
}

// AddAllocation remembers the node which a pod with claims got nominated to.
// A pod which got assumed is not a victim anymore, for example after it got
// preempted in memory while it waited in Permit.
func (pl *DynamicResources) AddAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations == nil {
		return framework.AllocationChangeResult{}
	}
	if pod.Spec.NodeName != "" {
		pl.nominations.forgetVictim(pod.UID)
	} else if len(pod.Spec.ResourceClaims) > 0 {
		pl.nominations.nominate(pod.UID, nodeName)
	}
	return framework.AllocationChangeResult{}
}

// RemoveAllocation forgets the nomination of the pod to the node.
func (pl *DynamicResources) RemoveAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations != nil && pod.Spec.NodeName == "" {
		pl.nominations.removeNomination(pod.UID, nodeName)
	}
	return framework.AllocationChangeResult{}
}
