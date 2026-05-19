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
	"bytes"
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// OpportunisticBatching caches results from filtering and scoring when possible to optimize
// scheduling of common pods.
type OpportunisticBatch struct {
	state         *batchState
	lastCycle     schedulingCycle
	signatureFunc SignatureFunc
	handle        fwk.Handle

	// Used primarily for tests, count the total pods we
	// have successfully batched.
	batchedPods int64
}

type batchState struct {
	signature    fwk.PodSignature
	sortedNodes  framework.SortedScoredNodes
	creationTime time.Time
}

type schedulingCycle struct {
	cycleCount int64
	chosenNode string
	succeeded  bool
}

const (
	maxBatchAge = 500 * time.Millisecond
)

type SignatureFunc func(h fwk.Handle, ctx context.Context, p *v1.Pod, recordPluginStats bool) fwk.PodSignature

func signUsingFramework(h fwk.Handle, ctx context.Context, p *v1.Pod, recordPluginStats bool) fwk.PodSignature {
	return h.SignPod(ctx, p, recordPluginStats)
}

// Provide a hint for the pod based on filtering an scoring results of previous cycles. Caching works only for consecutive pods
// with the same signature that are scheduled in 1-pod-per-node manner (otherwise previous scores could not be reused).
// It's assured by checking the top rated node is no longer feasible (meaning the previous pod was successfully scheduled and the
// current one does not fit).
func (b *OpportunisticBatch) GetNodeHint(ctx context.Context, pod *v1.Pod, state fwk.CycleState, cycleCount int64) (string, fwk.PodSignature) {
	logger := klog.FromContext(ctx)
	var hint string

	startTime := time.Now()
	defer func() {
		hinted := "hint"
		if hint == "" {
			hinted = "no_hint"
		}
		metrics.GetNodeHintDuration.WithLabelValues(hinted, b.handle.ProfileName()).Observe(metrics.SinceInSeconds(startTime))
	}()

	signature := b.signatureFunc(b.handle, ctx, pod, state.ShouldRecordPluginMetrics())

	nodeInfos := b.handle.SnapshotSharedLister().NodeInfos()

	// If we don't have state that we can use, then return an empty hint.
	if !b.batchStateCompatible(ctx, logger, pod, signature, cycleCount, state, nodeInfos) {
		metrics.BatchAttemptStats.WithLabelValues(b.handle.ProfileName(), metrics.BatchAttemptNoHint).Inc()
		logger.V(4).Info("OpportunisticBatch no node hint available for pod",
			"profile", b.handle.ProfileName(), "pod", klog.KObj(pod), "cycleCount", cycleCount)
		return "", signature
	}

	// Otherwise, pop the head of the list in our state and return it as
	// a hint. Also record it in our data to compare on storage.
	hint = b.state.sortedNodes.Pop()
	logger.V(3).Info("OpportunisticBatch provided node hint",
		"profile", b.handle.ProfileName(), "pod", klog.KObj(pod), "cycleCount", cycleCount, "hint", hint,
		"remainingNodes", b.state.sortedNodes.Len())

	return hint, signature
}

// Store results from scheduling for use as a batch later.
func (b *OpportunisticBatch) StoreScheduleResults(ctx context.Context, signature fwk.PodSignature, hintedNode, chosenNode string, otherNodes framework.SortedScoredNodes, cycleCount int64) {
	logger := klog.FromContext(ctx)

	startTime := time.Now()
	defer metrics.StoreScheduleResultsDuration.WithLabelValues(b.handle.ProfileName()).Observe(metrics.SinceInSeconds(startTime))

	// Set our cycle information for next time.
	b.lastCycle = schedulingCycle{
		cycleCount: cycleCount,
		chosenNode: chosenNode,
		succeeded:  true,
	}
	logger.V(4).Info("OpportunisticBatch set cycle state",
		"profile", b.handle.ProfileName(), "cycleCount", cycleCount, "hintedNode", hintedNode, "chosenNode", chosenNode)

	if hintedNode == chosenNode {
		logger.V(4).Info("OpportunisticBatch skipping set state because hint was provided",
			"profile", b.handle.ProfileName(), "cycleCount", cycleCount, "chosenNode", chosenNode)
		metrics.BatchAttemptStats.WithLabelValues(b.handle.ProfileName(), metrics.BatchAttemptHintUsed).Inc()
		b.batchedPods++
		return
	}

	// Only store new results if we didn't give a hint or it wasn't used.
	// Track the somewhat unusual case where we gave a hint and it wasn't used.
	if hintedNode != "" {
		metrics.BatchAttemptStats.WithLabelValues(b.handle.ProfileName(), metrics.BatchAttemptHintNotUsed).Inc()
		logger.V(4).Info("OpportunisticBatch hint not used",
			"profile", b.handle.ProfileName(), "cycleCount", b.lastCycle.cycleCount, "hint", hintedNode, "chosen", chosenNode)
	}

	// If this pod is batchable, set our results as state.
	// We will check this against the next pod when we give the
	// next hint.
	if signature != nil && otherNodes != nil && otherNodes.Len() > 0 {
		b.state = &batchState{
			signature:    signature,
			sortedNodes:  otherNodes,
			creationTime: time.Now(),
		}
		if loggerV := logger.V(6); loggerV.Enabled() {
			loggerV.Info("OpportunisticBatch set batch information",
				"profile", b.handle.ProfileName(), "signature", b.state.signature, "nodes", otherNodes.Len(), "cycleCount", cycleCount)
		} else {
			logger.V(4).Info("OpportunisticBatch set batch information",
				"profile", b.handle.ProfileName(), "nodes", otherNodes.Len(), "cycleCount", cycleCount)
		}
	} else {
		reason := metrics.BatchFlushPodNotBatchable
		if otherNodes == nil || otherNodes.Len() == 0 {
			reason = metrics.BatchFlushEmptyList
		}

		b.logUnusableState(logger, cycleCount, reason)
		b.state = nil
	}
}

// logUnusableState our batch state because we can't keep it up to date.
// Record the reason for our invalidation in the stats.
func (b *OpportunisticBatch) logUnusableState(logger klog.Logger, cycleCount int64, reason string) {
	metrics.BatchCacheFlushed.WithLabelValues(b.handle.ProfileName(), reason).Inc()
	logger.V(4).Info("OpportunisticBatch found unusable state",
		"profile", b.handle.ProfileName(), "cycleCount", cycleCount, "reason", reason)
}

// Update the batch state based on a the arrival of a new pod and the chosen node from the last pod.
func (b *OpportunisticBatch) batchStateCompatible(ctx context.Context, logger klog.Logger, pod *v1.Pod, signature fwk.PodSignature, cycleCount int64, state fwk.CycleState, nodeInfos fwk.NodeInfoLister) bool {
	// Just return if we don't have any state to use.
	if b.stateEmpty() {
		return false
	}

	// In this case, a previous pod was scheduled by another profile, meaning we can't use the state anymore.
	if cycleCount != b.lastCycle.cycleCount+1 {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushPodSkipped)
		return false
	}

	// If our last pod failed we can't use the state.
	if !b.lastCycle.succeeded {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushPodFailed)
		return false
	}

	// If the new pod is incompatible with the current state, throw the state away.
	if signature == nil || !bytes.Equal(signature, b.state.signature) {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushPodIncompatible)
		return false
	}

	// If the state is too old, throw the state away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.state.creationTime.Add(maxBatchAge))) {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushExpired)
		return false
	}

	// We can only reuse the previous state if the new pod will not
	// fit on the node we used for the last pod. If the node we
	// chose last time can't host this pod, then we know
	// that the next best should be the next node in the list.
	// If we *could* put this pod on the node we just used, then we can't
	// use our cache because we don't know how to rescore the used node; it
	// could be the best, or it could be somewhere else.
	// See git.k8s.io/enhancements/keps/sig-scheduling/5598-opportunistic-batching
	lastChosenNode, err := nodeInfos.Get(b.lastCycle.chosenNode)
	if lastChosenNode == nil || err != nil {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushNodeMissing)
		return false
	}

	status := b.handle.RunFilterPlugins(ctx, state, pod, lastChosenNode)
	if !status.IsRejected() {
		b.logUnusableState(logger, cycleCount, metrics.BatchFlushNodeNotFull)
		return false
	}

	// Our state matches with our new pod and is useable
	return true
}

// Irritatingly we can end up with a variety of different configurations that are all "empty".
// Rather than trying to normalize all cases when they happen, we just check them all.
func (b *OpportunisticBatch) stateEmpty() bool {
	return b.state == nil || b.state.sortedNodes == nil || b.state.sortedNodes.Len() == 0
}

func newOpportunisticBatch(h fwk.Handle, signatureFunc SignatureFunc) *OpportunisticBatch {
	return &OpportunisticBatch{
		signatureFunc: signatureFunc,
		handle:        h,
	}
}
