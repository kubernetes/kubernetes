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

package runtime

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// Batch is a plugin that reuses previous filtering / scoring results for pods when it is possible.
type OpportunisticBatch struct {
	state *batchState
}

type batchState struct {
	// signature        string
	sortedNodes  framework.SortedScoredNodes
	podSucceeded bool
	creationTime time.Time
}

const (
	maxBatchAge = 500 * time.Millisecond
)

// Invalidate our batch state because we can't keep it up to date.
// Record the reason for our invalidation in the stats.
func (b *OpportunisticBatch) invalidate(result, reason string) {
	if b.state != nil {
		metrics.BatchUsageStats.WithLabelValues(result, reason).Inc()
		b.state = nil
	}
}

// PreFilter invoked at the prefilter extension point.
func (b *OpportunisticBatch) NewPod(ctx context.Context, pod *v1.Pod) {
	// If our state is empty, then just return
	if b.state == nil {
		return
	}

	// If our list is empty then clear our state.
	if b.state.sortedNodes.Len() == 0 {
		b.invalidate("failure", "empty_list")
		return
	}

	// If our last pod didn't succeed, then clear our state
	if !b.state.podSucceeded {
		b.invalidate("failure", "pod_failure")
		return
	}

	// If this pod already has a nominated name, then we can't use the batch
	// (and don't need to!)
	if pod.Status.NominatedNodeName != "" {
		b.invalidate("failure", "already_nominated")
		return
	}

	// If the pod is incompatible with this batch, then we can't use it.
	if !b.podCompatible(pod) {
		b.invalidate("failure", "pod_incompatible")
		return
	}

	// If the batch is too old, throw it away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.state.creationTime.Add(maxBatchAge))) {
		b.invalidate("failure", "expired")
		return
	}

	// We have matching state for the given pod!
	// Narrow our viable nodes to just the single node
	// we want.
	metrics.BatchUsageStats.WithLabelValues("success", "success").Inc()

	// Clear the success bit; the NodeResults call will set it.
	b.state.podSucceeded = false
}

func (b *OpportunisticBatch) NodeHint(ctx context.Context, pod *v1.Pod) string {
	if b.state == nil {
		return ""
	}
	return b.state.sortedNodes.Pop()
}

func (b *OpportunisticBatch) podCompatible(_ *v1.Pod) bool {
	// In future PRs, we will use the signatures to determine compatibility.
	return false
}

func (b *OpportunisticBatch) postScore(ctx context.Context, inputState fwk.CycleState, thisFramework bool, owningFwk framework.Framework, podInfo fwk.PodInfo, inputChosenNode fwk.NodeInfo, otherNodes framework.SortedScoredNodes) {
	// A pod from another framework means we need to invalidate our results.
	if !thisFramework {
		b.invalidate("failure", "other_fwk_pod")
		return
	}

	if otherNodes == nil {
		b.invalidate("failure", "empty_list")
		return
	}

	pod := podInfo.GetPod().DeepCopy()
	state := inputState.Clone()
	chosenNode := inputChosenNode.Snapshot()

	// Update the state assuming placement of the previous pod. If we fail, throw away the batch.
	chosenNode.AddPodInfo(podInfo)
	status := owningFwk.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, chosenNode)
	if !status.IsSuccess() {
		b.invalidate("error", "add_pod_failed")
		return
	}

	// Now check if the node we used can be filtered. If it is no longer feasible,
	// then we can continue using the batch. Otherwise throw the batch away, because
	// we can't rescore the individual node.
	status = owningFwk.RunFilterPlugins(ctx, state, pod, chosenNode)
	if !status.IsRejected() {
		b.invalidate("failure", "node_not_filtered")
		return
	}

	// Fill the state with the results from our schedulePod call if it is empty.
	if b.state == nil {
		b.state = &batchState{
			sortedNodes:  otherNodes,
			creationTime: time.Now(),
		}
	}

	// Make sure we reuse our state for the next pod.
	b.state.podSucceeded = true
}
