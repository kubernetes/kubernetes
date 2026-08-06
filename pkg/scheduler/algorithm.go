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

package scheduler

import (
	"context"
	"errors"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

type SchedulingAlgorithm struct {
	nodeInfoSnapshot         *internalcache.Snapshot
	cache                    internalcache.Cache
	percentageOfNodesToScore int32
	cycleProvider            func() int64
	nextStartNodeIndex       int
	schedulePodOverride      func(ctx context.Context, f framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) (ScheduleResult, error)
	numNodesToFindOverride   func(numAllNodes int32) int32
}

func (a *SchedulingAlgorithm) currentCycle() int64 {
	if a.cycleProvider != nil {
		return a.cycleProvider()
	}
	return 0
}

type AlgorithmOption func(*SchedulingAlgorithm)

func WithNumNodesToFindOverride(fn func(numAllNodes int32) int32) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.numNodesToFindOverride = fn
	}
}
func WithCycleProvider(fn func() int64) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.cycleProvider = fn
	}
}

// WithCache attaches a scheduler cache, enabling AssumeAndReserveInCache and
// UnreserveAndForgetFromCache. An algorithm built without it can only assume
// into its snapshot: pods assumed there are dropped by the next UpdateSnapshot,
// whereas cache-assumed pods survive it until confirmed or forgotten.
func WithCache(cache internalcache.Cache) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.cache = cache
	}
}

// withPercentageOfNodesToScore is unexported because the exported name is taken by
// the scheduler.Option serving the same purpose, and Go has no
// overloading. kube-scheduler wiring lives in this package, so unexported is enough.
func withPercentageOfNodesToScore(percentage int32) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.percentageOfNodesToScore = percentage
	}
}

// NewSchedulingAlgorithm creates a standalone in-memory scheduling algorithm
// operating on the given snapshot. The algorithm assumes pods only where the caller
// asks it to: pass WithCache to enable the cache methods, otherwise only the snapshot
// methods are available.
func NewSchedulingAlgorithm(snapshot *internalcache.Snapshot, opts ...AlgorithmOption) *SchedulingAlgorithm {
	a := &SchedulingAlgorithm{
		nodeInfoSnapshot:         snapshot,
		percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// runSchedulePod dispatches to the test-only override if one was installed,
// otherwise to the algorithm's own implementation. Production always falls
// through to a.schedulePod. Scheduler.SchedulePod must call schedulePod directly
// rather than runSchedulePod, so an installed override cannot recurse through it.
func (a *SchedulingAlgorithm) runSchedulePod(ctx context.Context, f framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) (ScheduleResult, error) {
	if a.schedulePodOverride != nil {
		return a.schedulePodOverride(ctx, f, state, podInfo)
	}
	return a.schedulePod(ctx, f, state, podInfo)
}

// FindAllNodesThatFitPod evaluates all placement nodes without early-exit shortcuts
// so callers inspecting cluster capacity or running custom batching receive exhaustive results.
func (a *SchedulingAlgorithm) FindAllNodesThatFitPod(ctx context.Context, state fwk.CycleState, schedFramework framework.Framework, podInfo *framework.QueuedPodInfo) ([]fwk.NodeInfo, framework.Diagnosis, error) {
	nodes, diagnosis, _, err := a.findNodesThatFitPod(ctx, schedFramework, state, podInfo, true)
	return nodes, diagnosis, err
}

// AssumeAndReserveInCache assumes the pod into the scheduler cache and runs Reserve plugins.
// If Reserve plugins fail, the assumption is rolled back from the cache.
func (a *SchedulingAlgorithm) AssumeAndReserveInCache(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
	scheduleResult ScheduleResult) (*framework.QueuedPodInfo, *fwk.Status) {

	logger := klog.FromContext(ctx)
	host := scheduleResult.SuggestedHost
	if a.cache == nil {
		return podInfo, fwk.AsStatus(errors.New("SchedulingAlgorithm was built without a cache: " +
			"use WithCache, or assume into the snapshot instead"))
	}
	assumedPodInfo := a.prepareAssumedPod(logger, state, podInfo, host)
	if err := a.cache.AssumePod(logger, assumedPodInfo.Pod); err != nil {
		logger.Error(err, "Scheduler cache AssumePod failed")
		return assumedPodInfo, fwk.AsStatus(err)
	}
	schedFramework.DeleteNominatedPodIfExists(assumedPodInfo.Pod)

	if status := a.reserveOrUndo(ctx, state, schedFramework, assumedPodInfo, podInfo.Pod, host,
		func() error {
			return a.UnreserveAndForgetFromCache(ctx, state, schedFramework, assumedPodInfo, host)
		}); status != nil {
		return assumedPodInfo, status
	}
	return assumedPodInfo, nil
}

// UnreserveAndForgetFromCache runs Unreserve plugins and forgets the pod from the scheduler cache.
func (a *SchedulingAlgorithm) UnreserveAndForgetFromCache(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, assumedPodInfo *framework.QueuedPodInfo, nodeName string) error {

	logger := klog.FromContext(ctx)
	schedFramework.RunReservePluginsUnreserve(ctx, state, assumedPodInfo.Pod, nodeName)
	// No nomination restore here: a pod that fails after a cache assume goes back
	// through handleSchedulingFailure, which re-adds the nomination itself.
	return a.cache.ForgetPod(logger, assumedPodInfo.Pod)
}

// AssumeAndReserveInSnapshot assumes the pod into the transient node snapshot and runs Reserve plugins.
// Returns a revert function to roll back the assumption and reservation from the snapshot if needed.
func (a *SchedulingAlgorithm) AssumeAndReserveInSnapshot(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
	scheduleResult ScheduleResult) (*framework.QueuedPodInfo, *fwk.Status, func()) {

	logger := klog.FromContext(ctx)
	host := scheduleResult.SuggestedHost

	assumedPodInfo := a.prepareAssumedPod(logger, state, podInfo, host)
	if err := a.nodeInfoSnapshot.AssumePod(assumedPodInfo.PodInfo); err != nil {
		logger.Error(err, "Scheduler snapshot AssumePod failed")
		return assumedPodInfo, fwk.AsStatus(err), nil
	}
	schedFramework.DeleteNominatedPodIfExists(assumedPodInfo.Pod)

	revert := func() {
		if err := a.UnreserveAndForgetFromSnapshot(ctx, state, schedFramework, assumedPodInfo, host); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "ForgetPod failed")
		}
	}
	if status := a.reserveOrUndo(ctx, state, schedFramework, assumedPodInfo, podInfo.Pod, host,
		func() error {
			return a.UnreserveAndForgetFromSnapshot(ctx, state, schedFramework, assumedPodInfo, host)
		}); status != nil {
		return nil, status, nil
	}
	return assumedPodInfo, nil, revert
}

// UnreserveAndForgetFromSnapshot runs Unreserve plugins, forgets the pod from the node snapshot,
// and restores any existing pod nomination.
func (a *SchedulingAlgorithm) UnreserveAndForgetFromSnapshot(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, assumedPodInfo *framework.QueuedPodInfo, nodeName string) error {

	logger := klog.FromContext(ctx)
	schedFramework.RunReservePluginsUnreserve(ctx, state, assumedPodInfo.Pod, nodeName)
	if err := a.nodeInfoSnapshot.ForgetPod(logger, assumedPodInfo.Pod); err != nil {
		return err
	}
	if assumedPodInfo.Pod.Status.NominatedNodeName != "" {
		// The assume removed the nomination; reverting a tentative assume restores it.
		schedFramework.AddNominatedPod(logger, assumedPodInfo.PodInfo, &fwk.NominatingInfo{
			NominatedNodeName: assumedPodInfo.Pod.Status.NominatedNodeName,
			NominatingMode:    fwk.ModeOverride,
		})
	}
	return nil
}

// SchedulePod runs the filter and score phases for one pod and returns the selected
// host. It assumes nothing and runs no PostFilter: preemption is the caller's policy.
func (a *SchedulingAlgorithm) SchedulePod(ctx context.Context, schedFramework framework.Framework,
	state fwk.CycleState, podInfo *framework.QueuedPodInfo) (ScheduleResult, error) {
	return a.runSchedulePod(ctx, schedFramework, state, podInfo)
}

// ScheduleErrorToStatus maps an error from SchedulePod onto the Status
// kube-scheduler uses for it, so consumers classify failures identically.
func ScheduleErrorToStatus(err error) *fwk.Status {
	if err == ErrNoNodesAvailable {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable).WithError(err)
	}
	if _, ok := err.(*framework.FitError); !ok {
		return fwk.AsStatus(err)
	}
	return fwk.NewStatus(fwk.Unschedulable).WithError(err)
}
