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
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

type PodSignatureFunc func(p *v1.Pod) string

// Only needed until we connect signatures
func noBatchSignatures(p *v1.Pod) string { return "" }

// OpportunisticBatching caches results from filtering and scoring when possible to optimize
// scheudling of common pods.
type OpportunisticBatch struct {
	state         *batchState
	signatureFunc PodSignatureFunc
}

type batchState struct {
	signature    string
	sortedNodes  framework.SortedScoredNodes
	podSucceeded bool
	lastHint     string
	creationTime time.Time
}

const (
	maxBatchAge = 500 * time.Millisecond
)

// Invalidate our batch state because we can't keep it up to date.
// Record the reason for our invalidation in the stats.
func (b *OpportunisticBatch) invalidate(logger klog.Logger, result, reason string) {
	logger.V(2).Info("Invalidate called", "result", result, "reason", reason)
	if b.state != nil {
		metrics.BatchUsageStats.WithLabelValues(result, reason).Inc()
		b.state = nil
		logger.V(2).Info("Invalidate change", "result", result, "reason", reason)
	}
}

// PreFilter invoked at the prefilter extension point.
func (b *OpportunisticBatch) NewPod(ctx context.Context, pod *v1.Pod) {
	log := klog.FromContext(ctx)

	// If our state is empty, then just return
	if b.state == nil || b.state.sortedNodes == nil || b.state.sortedNodes.Len() == 0 {
		b.invalidate(log, "failure", "empty_list")
		return
	}

	// If our last pod didn't succeed, then clear our state
	if !b.state.podSucceeded {
		b.invalidate(log, "failure", "pod_failure")
		return
	}

	// If this pod already has a nominated name, then we can't use the batch
	// (and don't need to!)
	if pod.Status.NominatedNodeName != "" {
		b.invalidate(log, "failure", "already_nominated")
		return
	}

	signature := b.signatureFunc(pod)

	// If the pod is incompatible with this batch, then we can't use it.
	if !b.podCompatible(signature) {
		b.invalidate(log, "failure", "pod_incompatible")
		return
	}

	// If the batch is too old, throw it away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.state.creationTime.Add(maxBatchAge))) {
		b.invalidate(log, "failure", "expired")
		return
	}

	// We have matching state for the given pod!
	metrics.BatchUsageStats.WithLabelValues("success", "success").Inc()

	// Clear the success bit; the NodeResults call will set it.
	b.state.podSucceeded = false
}

func (b *OpportunisticBatch) NodeHint(ctx context.Context, pod *v1.Pod) string {
	if b.state == nil {
		return ""
	}
	hint := b.state.sortedNodes.Pop()
	b.state.lastHint = hint
	return hint
}

func (b *OpportunisticBatch) podCompatible(signature string) bool {
	return signature != "" && signature == b.state.signature
}

func (b *OpportunisticBatch) postScore(ctx context.Context, inputState fwk.CycleState, thisFramework bool, owningFwk framework.Framework, podInfo fwk.PodInfo, inputChosenNode fwk.NodeInfo, otherNodes framework.SortedScoredNodes) {
	log := klog.FromContext(ctx)

	// A pod from another framework means we need to invalidate our results.
	if !thisFramework {
		b.invalidate(log, "failure", "other_fwk_pod")
		return
	}

	pod := podInfo.GetPod()
	state := inputState.Clone()
	chosenNode := inputChosenNode.Snapshot()

	// If we have state, clear it if we didn't use the hint we provided.
	if b.state != nil && b.state.lastHint != chosenNode.Node().Name {
		b.invalidate(log, "failure", "different_node_used")
	}

	// Update the state assuming placement of the previous pod. If we fail, throw away the batch.
	chosenNode.AddPodInfo(podInfo)
	status := owningFwk.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, chosenNode)
	if !status.IsSuccess() {
		b.invalidate(log, "error", "add_pod_failed")
		return
	}

	// Now check if the node we used can be filtered. If it is no longer feasible,
	// then we can continue using the batch. Otherwise throw the batch away, because
	// we can't rescore the individual node.
	status = owningFwk.RunFilterPlugins(ctx, state, pod, chosenNode)
	if !status.IsRejected() {
		b.invalidate(log, "failure", "node_not_filtered")
		return
	}

	// We can use the state (new or existing)

	// Fill the state with the results from our schedulePod call if it is empty.
	if b.state == nil {
		b.state = &batchState{
			signature:    b.signatureFunc(pod),
			sortedNodes:  otherNodes,
			creationTime: time.Now(),
		}
		log.V(2).Info("Set batch info", "signature", b.state.signature)
	}

	// Make sure we reuse our state for the next pod.
	b.state.podSucceeded = true
	b.state.lastHint = ""
}

func newOpportunisticBatch(signatureFunc PodSignatureFunc) *OpportunisticBatch {
	return &OpportunisticBatch{signatureFunc: signatureFunc}
}
