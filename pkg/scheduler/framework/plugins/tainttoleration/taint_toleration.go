/*
Copyright 2019 The Kubernetes Authors.

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

package tainttoleration

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	v1helper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// TaintToleration is a plugin that checks if a pod tolerates a node's taints.
type TaintToleration struct {
	handle                                   fwk.Handle
	enableSchedulingQueueHint                bool
	enableTaintTolerationComparisonOperators bool
}

var _ fwk.FilterPlugin = &TaintToleration{}
var _ fwk.PreScorePlugin = &TaintToleration{}
var _ fwk.ScorePlugin = &TaintToleration{}
var _ fwk.EnqueueExtensions = &TaintToleration{}
var _ fwk.SignPlugin = &TaintToleration{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.TaintToleration
	// preScoreStateKey is the key in CycleState to TaintToleration pre-computed data for Scoring.
	preScoreStateKey = "PreScore" + Name
	// ErrReasonNotMatch is the Filter reason status when not matching.
	ErrReasonNotMatch = "node(s) had taints that the pod didn't tolerate"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *TaintToleration) Name() string {
	return Name
}

// Feasibility and scoring based on the pod's tolerations.
func (pl *TaintToleration) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	return []fwk.SignFragment{
		{Key: fwk.TolerationsSignerName, Value: fwk.TolerationsSigner(pod)},
	}, nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *TaintToleration) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	if pl.enableSchedulingQueueHint {
		return []fwk.ClusterEventWithHint{
			// When the QueueingHint feature is enabled, preCheck is eliminated and we don't need additional UpdateNodeLabel.
			{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.UpdateNodeTaint}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
			// When the QueueingHint feature is enabled,
			// the scheduling queue uses Pod/Update Queueing Hint
			// to determine whether a Pod's update makes the Pod schedulable or not.
			// https://github.com/kubernetes/kubernetes/pull/122234
			{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdatePodToleration}, QueueingHintFn: pl.isSchedulableAfterPodTolerationChange},
		}, nil
	}

	return []fwk.ClusterEventWithHint{
		// A note about UpdateNodeLabel event:
		// Ideally, it's supposed to register only Add | UpdateNodeTaint because UpdateNodeLabel will never change the result from this plugin.
		// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
		// See: https://github.com/kubernetes/kubernetes/issues/109437
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
		// No need to register the Pod event; the update to the unschedulable Pods already triggers the scheduling retry when QHint is disabled.
	}, nil
}

// isSchedulableAfterNodeChange is invoked for all node events reported by
// an informer. It checks whether that change made a previously unschedulable
// pod schedulable.
func (pl *TaintToleration) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalNode, modifiedNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	wasUntolerated := true
	if originalNode != nil {
		_, wasUntolerated = v1helper.FindMatchingUntoleratedTaint(logger, originalNode.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc(), pl.enableTaintTolerationComparisonOperators)
	}

	_, isUntolerated := v1helper.FindMatchingUntoleratedTaint(logger, modifiedNode.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc(), pl.enableTaintTolerationComparisonOperators)

	if wasUntolerated && !isUntolerated {
		logger.V(5).Info("node was created or updated, and this may make the Pod rejected by TaintToleration plugin in the previous scheduling cycle schedulable", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return fwk.Queue, nil
	}

	logger.V(5).Info("node was created or updated, but it doesn't change the TaintToleration plugin's decision", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
	return fwk.QueueSkip, nil
}

// Filter invoked at the filter extension point.
func (pl *TaintToleration) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()

	taint, isUntolerated := v1helper.FindMatchingUntoleratedTaint(logger, node.Spec.Taints, pod.Spec.Tolerations,
		helper.DoNotScheduleTaintsFilterFunc(),
		pl.enableTaintTolerationComparisonOperators)
	if !isUntolerated {
		return nil
	}

	klog.FromContext(ctx).V(4).Info("node had untolerated taints", "node", klog.KObj(node), "pod", klog.KObj(pod), "untoleratedTaint", taint)
	return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node(s) had untolerated taint(s)")
}

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	tolerationsPreferNoSchedule []v1.Toleration
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() fwk.StateData {
	return s
}

// getAllTolerationEffectPreferNoSchedule gets the list of all Tolerations with Effect PreferNoSchedule or with no effect.
func getAllTolerationPreferNoSchedule(tolerations []v1.Toleration) (tolerationList []v1.Toleration) {
	for _, toleration := range tolerations {
		// Empty effect means all effects which includes PreferNoSchedule, so we need to collect it as well.
		if len(toleration.Effect) == 0 || toleration.Effect == v1.TaintEffectPreferNoSchedule {
			tolerationList = append(tolerationList, toleration)
		}
	}
	return
}

// PreScore builds and writes cycle state used by Score and NormalizeScore.
func (pl *TaintToleration) PreScore(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	tolerationsPreferNoSchedule := getAllTolerationPreferNoSchedule(pod.Spec.Tolerations)
	state := &preScoreState{
		tolerationsPreferNoSchedule: tolerationsPreferNoSchedule,
	}
	cycleState.Write(preScoreStateKey, state)
	return nil
}

func getPreScoreState(cycleState fwk.CycleState) (*preScoreState, error) {
	c, err := cycleState.Read(preScoreStateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read %q from cycleState: %w", preScoreStateKey, err)
	}

	s, ok := c.(*preScoreState)
	if !ok {
		return nil, fmt.Errorf("%+v convert to tainttoleration.preScoreState error", c)
	}
	return s, nil
}

// CountIntolerableTaintsPreferNoSchedule gives the count of intolerable taints of a pod with effect PreferNoSchedule
func (pl *TaintToleration) countIntolerableTaintsPreferNoSchedule(logger klog.Logger, taints []v1.Taint, tolerations []v1.Toleration) (intolerableTaints int) {
	for _, taint := range taints {
		// check only on taints that have effect PreferNoSchedule
		if taint.Effect != v1.TaintEffectPreferNoSchedule {
			continue
		}

		if !v1helper.TolerationsTolerateTaint(logger, tolerations, &taint, pl.enableTaintTolerationComparisonOperators) {
			intolerableTaints++
		}
	}
	return
}

// Score invoked at the Score extension point.
func (pl *TaintToleration) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	logger := klog.FromContext(ctx)

	node := nodeInfo.Node()

	s, err := getPreScoreState(state)
	if err != nil {
		return 0, fwk.AsStatus(err)
	}

	score := int64(pl.countIntolerableTaintsPreferNoSchedule(logger, node.Spec.Taints, s.tolerationsPreferNoSchedule))
	return score, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *TaintToleration) NormalizeScore(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, scores fwk.NodeScoreList) *fwk.Status {
	return helper.DefaultNormalizeScore(fwk.MaxNodeScore, true, scores)
}

// ScoreExtensions of the Score plugin.
func (pl *TaintToleration) ScoreExtensions() fwk.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, h fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &TaintToleration{
		handle:                                   h,
		enableSchedulingQueueHint:                fts.EnableSchedulingQueueHint,
		enableTaintTolerationComparisonOperators: fts.EnableTaintTolerationComparisonOperators,
	}, nil
}

// isSchedulableAfterPodTolerationChange is invoked whenever a pod's toleration changed.
func (pl *TaintToleration) isSchedulableAfterPodTolerationChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.UID == modifiedPod.UID {
		// The updated Pod is the unschedulable Pod.
		logger.V(5).Info("a new toleration is added for the unschedulable Pod, and it may make it schedulable", "pod", klog.KObj(modifiedPod))
		return fwk.Queue, nil
	}

	logger.V(5).Info("a new toleration is added for a Pod, but it's an unrelated Pod and wouldn't change the TaintToleration plugin's decision", "pod", klog.KObj(modifiedPod))

	return fwk.QueueSkip, nil
}
