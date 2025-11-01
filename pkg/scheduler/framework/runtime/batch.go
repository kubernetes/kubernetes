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
// scheduling of common pods.
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
	pod        *v1.Pod
	succeeded  bool
	signature  string
	hint       string
	chosenNode string
}

const (
	maxBatchAge = 500 * time.Millisecond
)

// Provide a hint for the pod. Note that this should always be the same
// pod given in the preceding NewPod call, but if somehow this isn't the
// case we don't give a hint, which will then clear our state later.
func (b *OpportunisticBatch) RunNodeHint(ctx context.Context, pod *v1.Pod, state fwk.CycleState, lastHintedNode fwk.NodeInfo) string {
	logger := klog.FromContext(ctx)

	// If we don't have state that we can use, then return an empty hint.
	if !b.updateState(ctx, logger, pod, state, lastHintedNode) {
		metrics.BatchUsageStats.WithLabelValues("no_hint").Inc()
		logger.V(3).Info("Hint", "pod", pod.GetUID(), "value", "")
		return ""
	}

	// Otherwise, pop the head of the list in our state and return it as
	// a hint. Also record it in our data to compare at "PostScore".
	metrics.BatchUsageStats.WithLabelValues("hint").Inc()
	hint := b.state.sortedNodes.Pop()
	b.currPod.hint = hint
	logger.V(3).Info("Hint", "pod", pod.GetUID(), "value", hint)

	return hint
}

func (b *OpportunisticBatch) StoreScheduleResults(ctx context.Context, pod *v1.Pod, chosenNode string, otherNodes framework.SortedScoredNodes) {
	logger := klog.FromContext(ctx)

	// If we used the batch to assign this pod, we can keep using the state in the batch.
	if b.podBatched(pod, chosenNode) {
		metrics.BatchUsageStats.WithLabelValues("hint_used").Inc()
		b.currPod.chosenNode = chosenNode
	} else {
		// If this pod is batchable, set our results as state.
		// We will check this against the next pod when we give the
		// next hint.
		if b.currPod.signature != "" {
			b.state = &batchState{
				signature:    b.currPod.signature,
				sortedNodes:  otherNodes,
				creationTime: time.Now(),
			}
			logger.V(3).Info("Set batch info", "signature", b.state.signature)

			b.currPod.chosenNode = chosenNode
		} else {
			b.invalidate(logger, "pod_not_batchable")
		}
	}
}

func (b *OpportunisticBatch) PostScore(ctx context.Context, thisFramework bool, pod *v1.Pod) {
	if b.currPod.pod != pod || !thisFramework {
		b.invalidate(klog.FromContext(ctx), "diff_pod")
	} else {
		b.currPod.succeeded = true
	}
}

func (b *OpportunisticBatch) LastChosen() string {
	return b.currPod.chosenNode
}

// Invalidate our batch state because we can't keep it up to date.
// Record the reason for our invalidation in the stats.
func (b *OpportunisticBatch) invalidate(logger klog.Logger, reason string) {
	metrics.BatchEventStats.WithLabelValues("invalidate", reason).Inc()
	if b.state != nil {
		b.state = nil
		logger.V(5).Info("Invalidate change", "reason", reason)
	}
}

// Update the batch state based on a the arrival of a new pod and the chosen node from the last pod.
func (b *OpportunisticBatch) updateState(ctx context.Context, logger klog.Logger, pod *v1.Pod, state fwk.CycleState, lastHintedNode fwk.NodeInfo) bool {
	// Store our old pod.
	lastPod := b.currPod

	// Update to our current pod.
	b.setPod(pod)

	// Just return if we don't have any state to update
	if b.stateEmpty() {
		b.state = nil
		return false
	}

	// If our last pod didn't succeed, then clear our state.
	if !lastPod.succeeded {
		b.invalidate(logger, "pod_failure")
		return false
	}

	// If the new pod is incompatible with the current state, throw the state away.
	if b.currPod.signature == "" || b.currPod.signature != b.state.signature {
		b.invalidate(logger, "pod_incompatible")
		return false
	}

	// If the state is too old, throw the state away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.state.creationTime.Add(maxBatchAge))) {
		b.invalidate(logger, "expired")
		return false
	}

	// We can only reuse the previous state if the new pod will not
	// fit on the node we used for the last pod. If the node we
	// chose last time can't host this pod, then we know
	// that the next best should be the next node in the list.
	// If we *could* put this pod on the node we just used, then we can't
	// use our cache because we don't know how to rescore the used node; it
	// could be the best, or it could be somewhere else.
	if lastHintedNode == nil {
		b.invalidate(logger, "node_missing")
		return false
	}

	status := b.schedFramework.RunFilterPlugins(ctx, state, pod, lastHintedNode)
	if !status.IsRejected() {
		b.invalidate(logger, "node_not_full")
		return false
	}

	// Our state matches with our new pod and is useable
	return true
}

// We used the batch for the given pod iff:
// - the pod is for our framework
// - it matches the last pod we received in RunNodeHint
// - we gave a hint
// - the hint was the node we chose.
// In this case we can potentially continue using our existing batch state.
func (b *OpportunisticBatch) podBatched(pod *v1.Pod, chosenNode string) bool {
	return b.currPod.pod == pod && b.currPod.hint != "" && b.currPod.hint == chosenNode
}

// Irritatingly we can end up with a variety of different configurations that are all "empty".
// Rather than trying to normalize all cases when they happen, we just check them all.
func (b *OpportunisticBatch) stateEmpty() bool {
	return b.state == nil || b.state.sortedNodes == nil || b.state.sortedNodes.Len() == 0
}

// Set the current pod.
func (b *OpportunisticBatch) setPod(pod *v1.Pod) {
	b.currPod = batchPodInfo{
		pod:       pod,
		signature: b.signatureFunc(pod),
		succeeded: false,
	}
}

func newOpportunisticBatch(f framework.Framework, signatureFunc PodSignatureFunc) *OpportunisticBatch {
	return &OpportunisticBatch{
		signatureFunc:  signatureFunc,
		schedFramework: f,
	}
}
