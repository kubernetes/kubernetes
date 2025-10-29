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

const (
	maxBatchAge = 500 * time.Millisecond
)

type NodeScoreHeap []fwk.NodePluginScores

type batchState struct {
	// signature        string
	sortedNodes    framework.SortedScoredNodes
	lastPod        *v1.Pod
	lastPodSuccess bool
	lastHint       string
	creationTime   time.Time
}

func (b *batchState) NextPod(ctx context.Context, pod *v1.Pod) framework.Batch {
	// If our list is empty then just clear our state.
	if b.sortedNodes.Len() == 0 {
		metrics.BatchUsageStats.WithLabelValues("fail", "empty_list").Inc()
		return nil
	}

	// If this pod already has a nominated name, then we can't use the batch
	// (and don't need to!)
	if pod.Status.NominatedNodeName != "" {
		metrics.BatchUsageStats.WithLabelValues("fail", "already_nominated").Inc()
		return nil
	}

	// If the last scheduling cycle failed, then we throw away the batch info.
	// We do this now rather than on error to avoid having to catch all
	// error paths.
	if !b.lastPodSuccess {
		metrics.BatchUsageStats.WithLabelValues("fail", "last_use_failed").Inc()
		return nil
	}

	// If the pod is incompatible with this batch, then we can't use it.
	if !b.podCompatible(pod) {
		metrics.BatchUsageStats.WithLabelValues("fail", "pod_incompatible").Inc()
		return nil
	}

	// If the batch is too old, throw it away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((b.creationTime.Add(maxBatchAge))) {
		metrics.BatchUsageStats.WithLabelValues("fail", "expired").Inc()
		return nil
	}

	b.lastPod = pod
	b.lastPodSuccess = false

	return b
}

func (b *batchState) Hint(pod *v1.Pod) string {
	// This shouldn't happen, but just in case.
	if pod != b.lastPod {
		return ""
	}

	// We can use the batch; set nominated node name and make sure
	// we don't reuse the batch unless we succeed at scheduling.
	metrics.BatchUsageStats.WithLabelValues("success", "success").Inc()
	b.lastHint = b.sortedNodes.Pop()
	return b.lastHint
}

func (f *frameworkImpl) UpdateBatchPodSuccess(batch framework.Batch, ctx context.Context, podInfo fwk.PodInfo, state fwk.CycleState, nodeInfo fwk.NodeInfo, sortedNodes framework.SortedScoredNodes) framework.Batch {
	// Fill the state with the results from our schedulePod call if it is empty.
	if batch == nil {
		return &batchState{
			sortedNodes:    sortedNodes,
			lastPodSuccess: true,
			lastPod:        podInfo.GetPod(),
			creationTime:   time.Now(),
		}
	} else {
		// This shouldn't happen, but catch in case we got a hint but didn't use it.
		b := batch.(*batchState)
		if b.lastHint != nodeInfo.Node().Name {
			metrics.BatchUsageStats.WithLabelValues("error", "different_node_used").Inc()
			return nil
		}
	}

	b := batch.(*batchState)

	pod := podInfo.GetPod()

	// Update the state assuming placement of the previous pod. If we fail, throw away the batch.
	status := f.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, nodeInfo)
	if status.Code() == fwk.Error {
		metrics.BatchUsageStats.WithLabelValues("error", "add_pod_failed").Inc()
		return nil
	}

	// Now check if the node we used can be filtered. If it is no longer feasible,
	// then we can continue using the batch. Otherwise throw the batch away, because
	// we can't rescore the individual node.
	status = f.RunFilterPlugins(ctx, state, pod, nodeInfo)
	if !status.IsRejected() {
		metrics.BatchUsageStats.WithLabelValues("fail", "node_not_filtered").Inc()
		return nil
	}

	// Now remove the pod to get ready for binding. Again throw away if we fail.
	status = f.RunPreFilterExtensionRemovePod(ctx, state, pod, podInfo, nodeInfo)
	if status.Code() == fwk.Error {
		metrics.BatchUsageStats.WithLabelValues("error", "remove_pod_failed").Inc()
		return nil
	}

	// We succeeded at updating the batch! Mark it as feasible to use
	// for next pod.
	b.lastPodSuccess = true
	return batch
}

func (b *batchState) podCompatible(pod *v1.Pod) bool {
	// In future PRs, we will use the signatures to determine ccompatibility.
	return false
}
