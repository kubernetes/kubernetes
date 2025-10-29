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

package opportunisticbatch

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// Batch is a plugin that checks if a node has free ports for the requested pod ports.
type OpportunisticBatch struct {
	handle fwk.Handle
	state  *batchState
}

type batchState struct {
	// signature        string
	sortedNodes  fwk.SortedScoredNodes
	podSucceeded bool
	creationTime time.Time
}

var _ fwk.PreFilterPlugin = &OpportunisticBatch{}
var _ fwk.NodeResultsPlugin = &OpportunisticBatch{}
var _ fwk.EnqueueExtensions = &OpportunisticBatch{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.OpportunisticBatch

	maxBatchAge = 500 * time.Millisecond
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *OpportunisticBatch) Name() string {
	return Name
}

// Invalidate our batch state because we can't keep it up to date.
// Record the reason for our invalidation in the stats.
func (pl *OpportunisticBatch) invalidate(result, reason string) {
	if pl.state != nil {
		metrics.BatchUsageStats.WithLabelValues(result, reason).Inc()
		pl.state = nil
	}
}

// PreFilter invoked at the prefilter extension point.
func (pl *OpportunisticBatch) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	// If our state is empty, then just return
	if pl.state == nil {
		return nil, nil
	}

	// If our list is empty then clear our state.
	if pl.state.sortedNodes.Len() == 0 {
		pl.invalidate("failure", "empty_list")
		return nil, nil
	}

	// If our last pod didn't succeed, then clear our state
	if !pl.state.podSucceeded {
		pl.invalidate("failure", "pod_failure")
		return nil, nil
	}

	// If this pod already has a nominated name, then we can't use the batch
	// (and don't need to!)
	if pod.Status.NominatedNodeName != "" {
		pl.invalidate("failure", "already_nominated")
		return nil, nil
	}

	// If the pod is incompatible with this batch, then we can't use it.
	if !pl.podCompatible(pod) {
		pl.invalidate("failure", "pod_incompatible")
		return nil, nil
	}

	// If the batch is too old, throw it away. This is to avoid
	// cases where we either have huge numbers of compatible pods in a
	// row or we have a long wait between pods.
	if time.Now().After((pl.state.creationTime.Add(maxBatchAge))) {
		pl.invalidate("failure", "expired")
		return nil, nil
	}

	// We have matching state for the given pod!
	// Narrow our viable nodes to just the single node
	// we want.
	metrics.BatchUsageStats.WithLabelValues("success", "success").Inc()

	// Clear the success bit; the NodeResults call will set it.
	pl.state.podSucceeded = false

	nodeName := pl.state.sortedNodes.Pop()
	return &fwk.PreFilterResult{
		NodeNames: sets.New(nodeName),
	}, nil
}

// PreFilterExtensions do not exist for this plugin.
func (pl *OpportunisticBatch) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

func (pl *OpportunisticBatch) podCompatible(_ *v1.Pod) bool {
	// In future PRs, we will use the signatures to determine compatibility.
	return false
}

func (pl *OpportunisticBatch) NodeResults(ctx context.Context, state fwk.CycleState, thisFramework bool, podInfo fwk.PodInfo, chosenNode fwk.NodeInfo, otherNodes fwk.SortedScoredNodes) {
	pod := podInfo.GetPod()

	// A pod from another framework means we need to invalidate our results.
	if !thisFramework {
		pl.invalidate("failure", "other_fwk_pod")
		return
	}

	// Fill the state with the results from our schedulePod call if it is empty.
	if pl.state == nil {
		pl.state = &batchState{
			sortedNodes:  otherNodes,
			creationTime: time.Now(),
			podSucceeded: true,
		}
	}

	// Update the state assuming placement of the previous pod. If we fail, throw away the batch.
	status := pl.handle.RunPreFilterExtensionAddPod(ctx, state, pod, podInfo, chosenNode)
	if status.Code() == fwk.Error {
		pl.invalidate("error", "add_pod_failed")
		return
	}

	// Now check if the node we used can be filtered. If it is no longer feasible,
	// then we can continue using the batch. Otherwise throw the batch away, because
	// we can't rescore the individual node.
	status = pl.handle.RunFilterPlugins(ctx, state, pod, chosenNode)
	if !status.IsRejected() {
		pl.invalidate("failure", "node_not_filtered")
		return
	}

	// Now remove the pod to get ready for binding. Again throw away if we fail.
	status = pl.handle.RunPreFilterExtensionRemovePod(ctx, state, pod, podInfo, chosenNode)
	if status.Code() == fwk.Error {
		pl.invalidate("error", "remove_pod_failed")
		return
	}

	// Make sure we reuse our state for the next pod.
	pl.state.podSucceeded = true
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *OpportunisticBatch) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{}, nil
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &OpportunisticBatch{handle: handle}, nil
}
