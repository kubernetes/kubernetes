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

package scheduler

import (
	"context"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

type Batch struct {
	state *batchState
}

type batchState struct {
	signature        string
	sortedNodes      nodeScoreHeap
	lastUseSucceeded bool
}

func (b *Batch) nominateIfPossible(podInfo *framework.QueuedPodInfo) {
	pod := podInfo.GetPodInfo().GetPod()

	// If we don't have any batch state, there is nothing to do.
	if b.state == nil {
		return
	}

	// If this pod already has a nominated name, then we can't use the batch
	// (and don't need to!)
	if pod.Status.NominatedNodeName != "" {
		b.state = nil
		return
	}

	// If the last scheduling cycle failed, then we throw away the batch info.
	// We do this now rather than on error to avoid having to catch all
	// error paths.
	if !b.state.lastUseSucceeded {
		b.state = nil
		return
	}

	// If the pod is incompatible with this batch, then we can't use it.
	if !b.podCompatible(pod) {
		b.state = nil
		return
	}

	// The pod is compatible and we have state; grab the top entry on our list.
	nn, err := selectHost(&b.state.sortedNodes)

	// If our list is empty then just clear our state.
	if err != nil {
		b.state = nil
		return
	}

	// We can use the batch; set nominated node name and make sure
	// we don't reuse the batch unless we succeed at scheduling.
	pod.Status.NominatedNodeName = nn.Name
	b.state.lastUseSucceeded = false
}

func (b *Batch) updateOnSuccess(ctx context.Context, schedFwk framework.Framework, podInfo fwk.PodInfo, state fwk.CycleState, nodeInfo fwk.NodeInfo, sortedNodes nodeScoreHeap) {
	// Fill the state with the results from our schedulePod call if it is empty.
	if b.state == nil {
		b.state = &batchState{
			sortedNodes:      sortedNodes,
			lastUseSucceeded: true,
		}
		return
	}

	pod := podInfo.GetPod()

	// Update the state assuming placement of the previous pod. If we fail, throw away the batch.
	status := schedFwk.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, nodeInfo)
	if status.Code() == fwk.Error {
		b.state = nil
		return
	}

	// Now check if the node we used can be filtered. If it is no longer feasible,
	// then we can continue using the batch. Otherwise throw the batch away, because
	// we can't rescore the individual node.
	status = schedFwk.RunFilterPlugins(ctx, state, pod, nodeInfo)
	if !status.IsRejected() {
		b.state = nil
		return
	}

	// We succeeded at updating the batch! Mark it as feasible to use
	// for next pod.
	b.state.lastUseSucceeded = true
}

func (b *Batch) podCompatible(pod *v1.Pod) bool {
	// In future PRs, we will use the signatures to determine ccompatibility.
	return false
}
