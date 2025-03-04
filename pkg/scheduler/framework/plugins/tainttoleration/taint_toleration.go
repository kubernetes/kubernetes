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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// TaintToleration is a plugin that checks if a pod tolerates a node's taints.
type TaintToleration struct {
	handle                    framework.Handle
	enableSchedulingQueueHint bool
}

var _ framework.FilterPlugin = &TaintToleration{}
var _ framework.PreScorePlugin = &TaintToleration{}
var _ framework.ScorePlugin = &TaintToleration{}
var _ framework.EnqueueExtensions = &TaintToleration{}

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

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *TaintToleration) EventsToRegister(_ context.Context) ([]framework.ClusterEventWithHint, error) {
	if pl.enableSchedulingQueueHint {
		return []framework.ClusterEventWithHint{
			// When the QueueingHint feature is enabled, preCheck is eliminated and we don't need additional UpdateNodeLabel.
			{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeTaint}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
			// When the QueueingHint feature is enabled,
			// the scheduling queue uses Pod/Update Queueing Hint
			// to determine whether a Pod's update makes the Pod schedulable or not.
			// https://github.com/kubernetes/kubernetes/pull/122234
			{Event: framework.ClusterEvent{Resource: framework.Pod, ActionType: framework.UpdatePodToleration}, QueueingHintFn: pl.isSchedulableAfterPodTolerationChange},
		}, nil
	}

	return []framework.ClusterEventWithHint{
		// A note about UpdateNodeLabel event:
		// Ideally, it's supposed to register only Add | UpdateNodeTaint because UpdateNodeLabel will never change the result from this plugin.
		// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
		// See: https://github.com/kubernetes/kubernetes/issues/109437
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeTaint | framework.UpdateNodeLabel}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
		// No need to register the Pod event; the update to the unschedulable Pods already triggers the scheduling retry when QHint is disabled.
	}, nil
}

// isSchedulableAfterNodeChange is invoked for all node events reported by
// an informer. It checks whether that change made a previously unschedulable
// pod schedulable.
func (pl *TaintToleration) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	originalNode, modifiedNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	wasUntolerated := true
	if originalNode != nil {
		_, wasUntolerated = v1helper.FindMatchingUntoleratedTaint(originalNode.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc())
	}

	_, isUntolerated := v1helper.FindMatchingUntoleratedTaint(modifiedNode.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc())

	if wasUntolerated && !isUntolerated {
		logger.V(5).Info("node was created or updated, and this may make the Pod rejected by TaintToleration plugin in the previous scheduling cycle schedulable", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return framework.Queue, nil
	}

	logger.V(5).Info("node was created or updated, but it doesn't change the TaintToleration plugin's decision", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
	return framework.QueueSkip, nil
}

// Filter invoked at the filter extension point.
func (pl *TaintToleration) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()

	taint, isUntolerated := v1helper.FindMatchingUntoleratedTaint(node.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc())
	if !isUntolerated {
		return nil
	}

	errReason := fmt.Sprintf("node(s) had untolerated taint {%s: %s}", taint.Key, taint.Value)
	return framework.NewStatus(framework.UnschedulableAndUnresolvable, errReason)
}

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	tolerationsPreferNoSchedule []v1.Toleration
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
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
func (pl *TaintToleration) PreScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodes []*framework.NodeInfo) *framework.Status {
	tolerationsPreferNoSchedule := getAllTolerationPreferNoSchedule(pod.Spec.Tolerations)
	state := &preScoreState{
		tolerationsPreferNoSchedule: tolerationsPreferNoSchedule,
	}
	cycleState.Write(preScoreStateKey, state)
	return nil
}

func getPreScoreState(cycleState *framework.CycleState) (*preScoreState, error) {
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
func countIntolerableTaintsPreferNoSchedule(taints []v1.Taint, tolerations []v1.Toleration) (intolerableTaints int) {
	for _, taint := range taints {
		// check only on taints that have effect PreferNoSchedule
		if taint.Effect != v1.TaintEffectPreferNoSchedule {
			continue
		}

		if !v1helper.TolerationsTolerateTaint(tolerations, &taint) {
			intolerableTaints++
		}
	}
	return
}

// Score invoked at the Score extension point.
func (pl *TaintToleration) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) (int64, *framework.Status) {
	node := nodeInfo.Node()

	s, err := getPreScoreState(state)
	if err != nil {
		return 0, framework.AsStatus(err)
	}

	score := int64(countIntolerableTaintsPreferNoSchedule(node.Spec.Taints, s.tolerationsPreferNoSchedule))
	return score, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *TaintToleration) NormalizeScore(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	return helper.DefaultNormalizeScore(framework.MaxNodeScore, true, scores)
}

// ScoreExtensions of the Score plugin.
func (pl *TaintToleration) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, h framework.Handle, fts feature.Features) (framework.Plugin, error) {
	return &TaintToleration{
		handle:                    h,
		enableSchedulingQueueHint: fts.EnableSchedulingQueueHint,
	}, nil
}

// isSchedulableAfterPodTolerationChange is invoked whenever a pod's toleration changed.
func (pl *TaintToleration) isSchedulableAfterPodTolerationChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	_, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	if pod.UID == modifiedPod.UID {
		// The updated Pod is the unschedulable Pod.
		logger.V(5).Info("a new toleration is added for the unschedulable Pod, and it may make it schedulable", "pod", klog.KObj(modifiedPod))
		return framework.Queue, nil
	}

	logger.V(5).Info("a new toleration is added for a Pod, but it's an unrelated Pod and wouldn't change the TaintToleration plugin's decision", "pod", klog.KObj(modifiedPod))

	return framework.QueueSkip, nil
}
