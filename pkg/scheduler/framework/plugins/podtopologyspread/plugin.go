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

package podtopologyspread

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// ErrReasonConstraintsNotMatch is used for PodTopologySpread filter error.
	ErrReasonConstraintsNotMatch = "node(s) didn't match pod topology spread constraints"
	// ErrReasonNodeLabelNotMatch is used when the node doesn't hold the required label.
	ErrReasonNodeLabelNotMatch = ErrReasonConstraintsNotMatch + " (missing required label)"
)

var systemDefaultConstraints = []v1.TopologySpreadConstraint{
	{
		TopologyKey:       v1.LabelHostname,
		WhenUnsatisfiable: v1.ScheduleAnyway,
		MaxSkew:           3,
	},
	{
		TopologyKey:       v1.LabelTopologyZone,
		WhenUnsatisfiable: v1.ScheduleAnyway,
		MaxSkew:           5,
	},
}

// PodTopologySpread is a plugin that ensures pod's topologySpreadConstraints is satisfied.
type PodTopologySpread struct {
	systemDefaulted                              bool
	parallelizer                                 fwk.Parallelizer
	defaultConstraints                           []v1.TopologySpreadConstraint
	sharedLister                                 fwk.SharedLister
	services                                     corelisters.ServiceLister
	replicationCtrls                             corelisters.ReplicationControllerLister
	replicaSets                                  appslisters.ReplicaSetLister
	statefulSets                                 appslisters.StatefulSetLister
	enableNodeInclusionPolicyInPodTopologySpread bool
	enableMatchLabelKeysInPodTopologySpread      bool
	enableSchedulingQueueHint                    bool
	enableTaintTolerationComparisonOperators     bool
}

var _ fwk.PreFilterPlugin = &PodTopologySpread{}
var _ fwk.FilterPlugin = &PodTopologySpread{}
var _ fwk.PreScorePlugin = &PodTopologySpread{}
var _ fwk.ScorePlugin = &PodTopologySpread{}
var _ fwk.EnqueueExtensions = &PodTopologySpread{}
var _ fwk.SignPlugin = &PodTopologySpread{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = names.PodTopologySpread

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodTopologySpread) Name() string {
	return Name
}

// Pod topology spread is not localized to a pod and node, so we cannot
// sign pods that have topology spread constraints, either explicit or
// defaulted.
func (pl *PodTopologySpread) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	if len(pod.Spec.TopologySpreadConstraints) > 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "pods with topology constraints are not signable")
	}

	if len(pl.defaultConstraints) > 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "pods with default topology constraints are not signable")
	}

	return nil, nil
}

// New initializes a new plugin and returns it.
func New(_ context.Context, plArgs runtime.Object, h fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	args, err := getArgs(plArgs)
	if err != nil {
		return nil, err
	}
	if err := validation.ValidatePodTopologySpreadArgs(nil, &args); err != nil {
		return nil, err
	}
	pl := &PodTopologySpread{
		parallelizer:       h.Parallelizer(),
		sharedLister:       h.SnapshotSharedLister(),
		defaultConstraints: args.DefaultConstraints,
		enableNodeInclusionPolicyInPodTopologySpread: fts.EnableNodeInclusionPolicyInPodTopologySpread,
		enableMatchLabelKeysInPodTopologySpread:      fts.EnableMatchLabelKeysInPodTopologySpread,
		enableSchedulingQueueHint:                    fts.EnableSchedulingQueueHint,
		enableTaintTolerationComparisonOperators:     fts.EnableTaintTolerationComparisonOperators,
	}
	if args.DefaultingType == config.SystemDefaulting {
		pl.defaultConstraints = systemDefaultConstraints
		pl.systemDefaulted = true
	}
	if len(pl.defaultConstraints) != 0 {
		if h.SharedInformerFactory() == nil {
			return nil, fmt.Errorf("SharedInformerFactory is nil")
		}
		pl.setListers(h.SharedInformerFactory())
	}
	return pl, nil
}

func getArgs(obj runtime.Object) (config.PodTopologySpreadArgs, error) {
	ptr, ok := obj.(*config.PodTopologySpreadArgs)
	if !ok {
		return config.PodTopologySpreadArgs{}, fmt.Errorf("want args to be of type PodTopologySpreadArgs, got %T", obj)
	}
	return *ptr, nil
}

func (pl *PodTopologySpread) setListers(factory informers.SharedInformerFactory) {
	pl.services = factory.Core().V1().Services().Lister()
	pl.replicationCtrls = factory.Core().V1().ReplicationControllers().Lister()
	pl.replicaSets = factory.Apps().V1().ReplicaSets().Lister()
	pl.statefulSets = factory.Apps().V1().StatefulSets().Lister()
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *PodTopologySpread) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	podActionType := fwk.Add | fwk.UpdatePodLabel | fwk.Delete
	if pl.enableSchedulingQueueHint {
		// When the QueueingHint feature is enabled, the scheduling queue uses Pod/Update Queueing Hint
		// to determine whether a Pod's update makes the Pod schedulable or not.
		// https://github.com/kubernetes/kubernetes/pull/122234
		// (If not, the scheduling queue always retries the unschedulable Pods when they're updated.)
		//
		// The Pod rejected by this plugin can be schedulable when the Pod has a spread constraint with NodeTaintsPolicy:Honor
		// and has got a new toleration.
		// So, we add UpdatePodToleration here only when QHint is enabled.
		podActionType = fwk.Add | fwk.UpdatePodLabel | fwk.UpdatePodToleration | fwk.Delete
	}

	return []fwk.ClusterEventWithHint{
		// All ActionType includes the following events:
		// - Add. An unschedulable Pod may fail due to violating topology spread constraints,
		// adding an assigned Pod may make it schedulable.
		// - UpdatePodLabel. Updating on an existing Pod's labels (e.g., removal) may make
		// an unschedulable Pod schedulable.
		// - Delete. An unschedulable Pod may fail due to violating an existing Pod's topology spread constraints,
		// deleting an existing Pod may make it schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: podActionType}, QueueingHintFn: pl.isSchedulableAfterPodChange},
		// Node add|delete|update maybe lead an topology key changed,
		// and make these pod in scheduling schedulable or unschedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.Delete | fwk.UpdateNodeLabel | fwk.UpdateNodeTaint}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
	}, nil
}

// involvedInTopologySpreading returns true if the incomingPod is involved in the topology spreading of podWithSpreading.
func involvedInTopologySpreading(incomingPod, podWithSpreading *v1.Pod) bool {
	return incomingPod.UID == podWithSpreading.UID ||
		(incomingPod.Spec.NodeName != "" && incomingPod.Namespace == podWithSpreading.Namespace)
}

// hasConstraintWithNodeTaintsPolicyHonor returns true if any constraint has `NodeTaintsPolicy: Honor`.
func hasConstraintWithNodeTaintsPolicyHonor(constraints []topologySpreadConstraint) bool {
	for _, c := range constraints {
		if c.NodeTaintsPolicy == v1.NodeInclusionPolicyHonor {
			return true
		}
	}
	return false
}

func (pl *PodTopologySpread) isSchedulableAfterPodChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalPod, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if (modifiedPod != nil && !involvedInTopologySpreading(modifiedPod, pod)) || (originalPod != nil && !involvedInTopologySpreading(originalPod, pod)) {
		logger.V(5).Info("the added/updated/deleted pod is unscheduled or has different namespace with target pod, so it doesn't make the target pod schedulable",
			"pod", klog.KObj(pod), "originalPod", klog.KObj(originalPod))
		return fwk.QueueSkip, nil
	}

	constraints, err := pl.getConstraints(pod)
	if err != nil {
		return fwk.Queue, err
	}

	// Pod is modified. Return Queue when the label(s) matching topologySpread's selector is added, changed, or deleted.
	if modifiedPod != nil && originalPod != nil {
		if pod.UID == modifiedPod.UID && !equality.Semantic.DeepEqual(modifiedPod.Spec.Tolerations, originalPod.Spec.Tolerations) && hasConstraintWithNodeTaintsPolicyHonor(constraints) {
			// If any constraint has `NodeTaintsPolicy: Honor`, we can return Queue when the target Pod has got a new toleration.
			logger.V(5).Info("the unschedulable pod has got a new toleration, which could make it schedulable",
				"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
			return fwk.Queue, nil
		}

		if equality.Semantic.DeepEqual(modifiedPod.Labels, originalPod.Labels) {
			logger.V(5).Info("the pod's update doesn't include the label update, which doesn't make the target pod schedulable",
				"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
			return fwk.QueueSkip, nil
		}
		for _, c := range constraints {
			if c.Selector.Matches(labels.Set(originalPod.Labels)) != c.Selector.Matches(labels.Set(modifiedPod.Labels)) {
				// This modification makes this Pod match(or not match) with this constraint.
				// Maybe now the scheduling result of topology spread gets changed by this change.
				logger.V(5).Info("a scheduled pod's label was updated and it makes the updated pod match or unmatch the pod's topology spread constraints",
					"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
				return fwk.Queue, nil
			}
		}
		// This modification of labels doesn't change whether this Pod would match selector or not in any constraints.
		logger.V(5).Info("a scheduled pod's label was updated, but it's a change unrelated to the pod's topology spread constraints",
			"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return fwk.QueueSkip, nil
	}

	// Pod is added. Return Queue when the added Pod has a label that matches with topologySpread's selector.
	if modifiedPod != nil {
		if podLabelsMatchSpreadConstraints(constraints, modifiedPod.Labels) {
			logger.V(5).Info("a scheduled pod was created and it matches with the pod's topology spread constraints",
				"pod", klog.KObj(pod), "createdPod", klog.KObj(modifiedPod))
			return fwk.Queue, nil
		}
		logger.V(5).Info("a scheduled pod was created, but it doesn't matches with the pod's topology spread constraints",
			"pod", klog.KObj(pod), "createdPod", klog.KObj(modifiedPod))
		return fwk.QueueSkip, nil
	}

	// Pod is deleted. Return Queue when the deleted Pod has a label that matches with topologySpread's selector.
	if podLabelsMatchSpreadConstraints(constraints, originalPod.Labels) {
		logger.V(5).Info("a scheduled pod which matches with the pod's topology spread constraints was deleted, and the pod may be schedulable now",
			"pod", klog.KObj(pod), "deletedPod", klog.KObj(originalPod))
		return fwk.Queue, nil
	}
	logger.V(5).Info("a scheduled pod was deleted, but it's unrelated to the pod's topology spread constraints",
		"pod", klog.KObj(pod), "deletedPod", klog.KObj(originalPod))

	return fwk.QueueSkip, nil
}

// getConstraints extracts topologySpreadConstraint(s) from the Pod spec.
// If the Pod doesn't have any topologySpreadConstraint, it returns default constraints.
func (pl *PodTopologySpread) getConstraints(pod *v1.Pod) ([]topologySpreadConstraint, error) {
	var constraints []topologySpreadConstraint
	var err error
	if len(pod.Spec.TopologySpreadConstraints) > 0 {
		// We have feature gating in APIServer to strip the spec
		// so don't need to re-check feature gate, just check length of Constraints.
		constraints, err = pl.filterTopologySpreadConstraints(
			pod.Spec.TopologySpreadConstraints,
			pod.Labels,
			v1.DoNotSchedule,
		)
		if err != nil {
			return nil, fmt.Errorf("obtaining pod's hard topology spread constraints: %w", err)
		}
	} else {
		constraints, err = pl.buildDefaultConstraints(pod, v1.DoNotSchedule)
		if err != nil {
			return nil, fmt.Errorf("setting default hard topology spread constraints: %w", err)
		}
	}
	return constraints, nil
}

// isSchedulableAfterNodeChange returns Queue when node has topologyKey in its labels, else return QueueSkip.
func (pl *PodTopologySpread) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalNode, modifiedNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	constraints, err := pl.getConstraints(pod)
	if err != nil {
		return fwk.Queue, err
	}

	var originalNodeMatching, modifiedNodeMatching bool
	if originalNode != nil {
		originalNodeMatching = nodeLabelsMatchSpreadConstraints(originalNode.Labels, constraints)
	}
	if modifiedNode != nil {
		modifiedNodeMatching = nodeLabelsMatchSpreadConstraints(modifiedNode.Labels, constraints)
	}

	// We return Queue in the following cases:
	// 1. Node/UpdateNodeLabel:
	// - The original node matched the pod's topology spread constraints, but the modified node does not.
	// - The modified node matches the pod's topology spread constraints, but the original node does not.
	// - The modified node matches the pod's topology spread constraints, and the original node and the modified node have different label values for any topologyKey.
	// 2. Node/UpdateNodeTaint:
	//  - The modified node match the pod's topology spread constraints, and the original node and the modified node have different taints.
	// 3. Node/Add: The created node matches the pod's topology spread constraints.
	// 4. Node/Delete: The original node matched the pod's topology spread constraints.
	if originalNode != nil && modifiedNode != nil {
		if originalNodeMatching != modifiedNodeMatching {
			logger.V(5).Info("the node is updated and now pod topology spread constraints has changed, and the pod may be schedulable now",
				"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode), "originalMatching", originalNodeMatching, "newMatching", modifiedNodeMatching)
			return fwk.Queue, nil
		}
		if modifiedNodeMatching && (checkTopologyKeyLabelsChanged(originalNode.Labels, modifiedNode.Labels, constraints) || !equality.Semantic.DeepEqual(originalNode.Spec.Taints, modifiedNode.Spec.Taints)) {
			logger.V(5).Info("the node is updated and now has different taints or labels, and the pod may be schedulable now",
				"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, nil
	}

	if modifiedNode != nil {
		if !modifiedNodeMatching {
			logger.V(5).Info("the created node doesn't match pod topology spread constraints",
				"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
			return fwk.QueueSkip, nil
		}
		logger.V(5).Info("the created node matches topology spread constraints, and the pod may be schedulable now",
			"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return fwk.Queue, nil
	}

	if !originalNodeMatching {
		logger.V(5).Info("the deleted node doesn't match pod topology spread constraints", "pod", klog.KObj(pod), "node", klog.KObj(originalNode))
		return fwk.QueueSkip, nil
	}
	logger.V(5).Info("the deleted node matches topology spread constraints, and the pod may be schedulable now",
		"pod", klog.KObj(pod), "node", klog.KObj(originalNode))
	return fwk.Queue, nil
}

// checkTopologyKeyLabelsChanged checks if any of the labels specified as topologyKey in the constraints have changed.
func checkTopologyKeyLabelsChanged(originalLabels, modifiedLabels map[string]string, constraints []topologySpreadConstraint) bool {
	for _, constraint := range constraints {
		topologyKey := constraint.TopologyKey
		if originalLabels[topologyKey] != modifiedLabels[topologyKey] {
			return true
		}
	}
	return false
}
