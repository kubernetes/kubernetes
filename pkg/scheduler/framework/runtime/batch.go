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
	state          *batchState
	currPod        batchPodInfo
	signatureFunc  PodSignatureFunc
	schedFramework framework.Framework
}

type batchState struct {
	signature    string
	sortedNodes  framework.SortedScoredNodes
	creationTime time.Time
}

type batchPodInfo struct {
	pod       *v1.Pod
	succeeded bool
	signed    bool
	signature string
	hint      string
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

// Inform the batch of a new pod. Note that this will only be called for
// pods that match this framework, while the "postScoring" call will
// happen for every pod that comes through the scheduler.
func (b *OpportunisticBatch) NewPod(ctx context.Context, pod *v1.Pod) {
	log := klog.FromContext(ctx)

	// If our last pod didn't succeed, then clear our state, but
	// keep going.
	if !b.currPod.succeeded {
		b.invalidate(log, "failure", "pod_failure")
	}

	// Update our current pod and check if the new pod is batchable.
	// Clear state if it isn't.
	if !b.setPod(pod) {
		b.invalidate(log, "failure", "pod_not_batchable")
		return
	}

	// Just return if we don't have any state.
	if b.stateEmpty() {
		b.state = nil
		return
	}

	// If the pod is incompatible with the state, throw the state away.
	if b.currPod.signature != b.state.signature {
		b.invalidate(log, "failure", "pod_incompatible")
		return
	}

	// If the state is too old, throw the state away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.state.creationTime.Add(maxBatchAge))) {
		b.invalidate(log, "failure", "expired")
		return
	}

	// Our state matches with our new pod.
	metrics.BatchUsageStats.WithLabelValues("success", "success").Inc()
}

// Provide a hint for the pod. Note that this should always be the same
// pod given in the preceeding NewPod call, but if somehow this isn't the
// case we don't give a hint, which will then clear our state later.
func (b *OpportunisticBatch) NodeHint(ctx context.Context, pod *v1.Pod) string {
	if b.stateEmpty() || b.currPod.pod != pod {
		return ""
	}
	hint := b.state.sortedNodes.Pop()
	b.currPod.hint = hint
	return hint
}

func (b *OpportunisticBatch) postScore(ctx context.Context, state fwk.CycleState, thisFramework bool, podInfo fwk.PodInfo, chosenNode fwk.NodeInfo, otherNodes framework.SortedScoredNodes) {
	log := klog.FromContext(ctx)
	pod := podInfo.GetPod()

	// If we know that we can't schedule any more pods of this signature on the node we just used,
	// then we can continue to use the additional scheduling information, either from this pod
	// or from the batch we used for this pod. This is because we know it is no longer feasible,
	// so the remaining nodes should remain in the same score order.
	if b.chosenNodeFull(ctx, state, podInfo, chosenNode) {
		// If we used the batch to assign this pod, and we still have results, we can
		// potentially keep using the remaining state for the next pod as well
		if b.podBatched(thisFramework, pod, chosenNode) && !b.stateEmpty() {
			b.currPod.succeeded = true
		} else {
			// Otherwise, check if we can use the results from this new pod.
			if b.setPod(pod) {
				// If so, set our state to try for next time.
				b.state = &batchState{
					signature:    b.currPod.signature,
					sortedNodes:  otherNodes,
					creationTime: time.Now(),
				}
				log.V(2).Info("Set batch info", "signature", b.state.signature)

				b.currPod.succeeded = true
			} else {
				b.invalidate(log, "failure", "pod_not_batchable")
			}
		}
	} else {
		b.invalidate(log, "failure", "node_not_full")
	}
}

// We used the batch for the given node if:
// - the pod is for our framework
// - it matches the last NewNode we recieved
// - we gave a hint
// - the hint was the node we chose.
// In this case we can potentially continue using our existing batch state.
func (b *OpportunisticBatch) podBatched(thisFramework bool, pod *v1.Pod, chosenNode fwk.NodeInfo) bool {
	return thisFramework && b.currPod.pod == pod && b.currPod.hint != "" && b.currPod.hint == chosenNode.Node().Name
}

// Irritatingly we can end up with a variety of different configurations that are all "empty".
// Rather than trying to normalize all cases when they happen, we just check them all.
func (b *OpportunisticBatch) stateEmpty() bool {
	return b.state == nil || b.state.sortedNodes == nil || b.state.sortedNodes.Len() == 0
}

// Check if the given node is "full", i.e. cannot host any more pods with this signature.
// We do this by assuming the current pod is already scheduled and calling the filter command.
func (b *OpportunisticBatch) chosenNodeFull(ctx context.Context, inputState fwk.CycleState, podInfo fwk.PodInfo, inputChosenNode fwk.NodeInfo) bool {
	pod := podInfo.GetPod()

	// Clone state to ensure we don't mess up the rest of the pipeline.
	state := inputState.Clone()
	chosenNode := inputChosenNode.Snapshot()

	chosenNode.AddPodInfo(podInfo)
	status := b.schedFramework.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, chosenNode)
	if !status.IsSuccess() {
		return false
	}

	status = b.schedFramework.RunFilterPlugins(ctx, state, pod, chosenNode)
	if !status.IsRejected() {
		return false
	}

	return true
}

// Set the current pod to match a new pod. Returns true
// if the new pod is batchable, false otherwise.
func (b *OpportunisticBatch) setPod(pod *v1.Pod) bool {
	b.currPod = batchPodInfo{
		pod:       pod,
		signature: b.signatureFunc(pod),
		succeeded: false,
	}
	return b.currPod.signature != ""
}

func newOpportunisticBatch(f framework.Framework, signatureFunc PodSignatureFunc) *OpportunisticBatch {
	return &OpportunisticBatch{
		signatureFunc:  signatureFunc,
		schedFramework: f,
	}
}
