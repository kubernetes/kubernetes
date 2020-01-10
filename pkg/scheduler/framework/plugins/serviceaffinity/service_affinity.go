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

package serviceaffinity

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "ServiceAffinity"

	// preFilterStateKey is the key in CycleState to ServiceAffinity pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// ErrReason is used for CheckServiceAffinity predicate error.
	ErrReason = "node(s) didn't match service affinity"
)

// Args holds the args that are used to configure the plugin.
type Args struct {
	// Labels are homogeneous for pods that are scheduled to a node.
	// (i.e. it returns true IFF this pod can be added to this node such that all other pods in
	// the same service are running on nodes with the exact same values for Labels).
	AffinityLabels []string `json:"labels,omitempty"`
	// AntiAffinityLabelsPreference are the labels to consider for service anti affinity scoring.
	AntiAffinityLabelsPreference []string `json:"antiAffinityLabelsPreference,omitempty"`
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	matchingPodList     []*v1.Pod
	matchingPodServices []*v1.Service
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	if s == nil {
		return nil
	}

	copy := preFilterState{}
	copy.matchingPodServices = append([]*v1.Service(nil),
		s.matchingPodServices...)
	copy.matchingPodList = append([]*v1.Pod(nil),
		s.matchingPodList...)

	return &copy
}

// New initializes a new plugin and returns it.
func New(plArgs *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	args := Args{}
	if err := framework.DecodeInto(plArgs, &args); err != nil {
		return nil, err
	}
	informerFactory := handle.SharedInformerFactory()
	serviceLister := informerFactory.Core().V1().Services().Lister()

	return &ServiceAffinity{
		sharedLister:  handle.SnapshotSharedLister(),
		serviceLister: serviceLister,
		args:          args,
	}, nil
}

// ServiceAffinity is a plugin that checks service affinity.
type ServiceAffinity struct {
	args          Args
	sharedLister  schedulerlisters.SharedLister
	serviceLister corelisters.ServiceLister
}

var _ framework.PreFilterPlugin = &ServiceAffinity{}
var _ framework.FilterPlugin = &ServiceAffinity{}
var _ framework.ScorePlugin = &ServiceAffinity{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *ServiceAffinity) Name() string {
	return Name
}

func (pl *ServiceAffinity) createPreFilterState(pod *v1.Pod) (*preFilterState, error) {
	if pod == nil {
		return nil, fmt.Errorf("a pod is required to calculate service affinity preFilterState")
	}
	// Store services which match the pod.
	matchingPodServices, err := schedulerlisters.GetPodServices(pl.serviceLister, pod)
	if err != nil {
		return nil, fmt.Errorf("listing pod services: %v", err.Error())
	}
	selector := createSelectorFromLabels(pod.Labels)
	allMatches, err := pl.sharedLister.Pods().List(selector)
	if err != nil {
		return nil, fmt.Errorf("listing pods: %v", err.Error())
	}

	// consider only the pods that belong to the same namespace
	matchingPodList := filterPodsByNamespace(allMatches, pod.Namespace)

	return &preFilterState{
		matchingPodList:     matchingPodList,
		matchingPodServices: matchingPodServices,
	}, nil
}

// PreFilter invoked at the prefilter extension point.
func (pl *ServiceAffinity) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	s, err := pl.createPreFilterState(pod)
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("could not create preFilterState: %v", err))

	}
	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *ServiceAffinity) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *ServiceAffinity) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	// If addedPod is in the same namespace as the pod, update the list
	// of matching pods if applicable.
	if s == nil || podToAdd.Namespace != podToSchedule.Namespace {
		return nil
	}

	selector := createSelectorFromLabels(podToSchedule.Labels)
	if selector.Matches(labels.Set(podToAdd.Labels)) {
		s.matchingPodList = append(s.matchingPodList, podToAdd)
	}

	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *ServiceAffinity) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if s == nil ||
		len(s.matchingPodList) == 0 ||
		podToRemove.Namespace != s.matchingPodList[0].Namespace {
		return nil
	}

	for i, pod := range s.matchingPodList {
		if pod.Name == podToRemove.Name && pod.Namespace == podToRemove.Namespace {
			s.matchingPodList = append(s.matchingPodList[:i], s.matchingPodList[i+1:]...)
			break
		}
	}

	return nil
}

func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// The metadata wasn't pre-computed in prefilter. We ignore the error for now since
		// Filter is able to handle that by computing it again.
		klog.V(5).Infof(fmt.Sprintf("reading %q from cycleState: %v", preFilterStateKey, err))
		return nil, nil
	}

	if c == nil {
		return nil, nil
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to interpodaffinity.state error", c)
	}
	return s, nil
}

// Filter matches nodes in such a way to force that
// ServiceAffinity.labels are homogeneous for pods that are scheduled to a node.
// (i.e. it returns true IFF this pod can be added to this node such that all other pods in
// the same service are running on nodes with the exact same ServiceAffinity.label values).
//
// For example:
// If the first pod of a service was scheduled to a node with label "region=foo",
// all the other subsequent pods belong to the same service will be schedule on
// nodes with the same "region=foo" label.
//
// Details:
//
// If (the svc affinity labels are not a subset of pod's label selectors )
// 	The pod has all information necessary to check affinity, the pod's label selector is sufficient to calculate
// 	the match.
// Otherwise:
// 	Create an "implicit selector" which guarantees pods will land on nodes with similar values
// 	for the affinity labels.
//
// 	To do this, we "reverse engineer" a selector by introspecting existing pods running under the same service+namespace.
//	These backfilled labels in the selector "L" are defined like so:
// 		- L is a label that the ServiceAffinity object needs as a matching constraint.
// 		- L is not defined in the pod itself already.
// 		- and SOME pod, from a service, in the same namespace, ALREADY scheduled onto a node, has a matching value.
func (pl *ServiceAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	if len(pl.args.AffinityLabels) == 0 {
		return nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}

	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	if s == nil {
		// Make the filter resilient in case preFilterState is missing.
		s, err = pl.createPreFilterState(pod)
		if err != nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("could not create preFilterState: %v", err))

		}
	}

	pods, services := s.matchingPodList, s.matchingPodServices
	filteredPods := nodeInfo.FilterOutPods(pods)
	// check if the pod being scheduled has the affinity labels specified in its NodeSelector
	affinityLabels := findLabelsInSet(pl.args.AffinityLabels, labels.Set(pod.Spec.NodeSelector))
	// Step 1: If we don't have all constraints, introspect nodes to find the missing constraints.
	if len(pl.args.AffinityLabels) > len(affinityLabels) {
		if len(services) > 0 {
			if len(filteredPods) > 0 {
				nodeWithAffinityLabels, err := pl.sharedLister.NodeInfos().Get(filteredPods[0].Spec.NodeName)
				if err != nil {
					return framework.NewStatus(framework.Error, "node not found")
				}
				addUnsetLabelsToMap(affinityLabels, pl.args.AffinityLabels, labels.Set(nodeWithAffinityLabels.Node().Labels))
			}
		}
	}
	// Step 2: Finally complete the affinity predicate based on whatever set of predicates we were able to find.
	if createSelectorFromLabels(affinityLabels).Matches(labels.Set(node.Labels)) {
		return nil
	}

	return framework.NewStatus(framework.Unschedulable, ErrReason)
}

// Score invoked at the Score extension point.
func (pl *ServiceAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.sharedLister.NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("node not found"))
	}

	// Pods matched namespace,selector on current node.
	var selector labels.Selector
	if services, err := schedulerlisters.GetPodServices(pl.serviceLister, pod); err == nil && len(services) > 0 {
		selector = labels.SelectorFromSet(services[0].Spec.Selector)
	} else {
		selector = labels.NewSelector()
	}

	if len(nodeInfo.Pods()) == 0 || selector.Empty() {
		return 0, nil
	}
	var score int64
	for _, existingPod := range nodeInfo.Pods() {
		// Ignore pods being deleted for spreading purposes
		// Similar to how it is done for SelectorSpreadPriority
		if pod.Namespace == existingPod.Namespace && existingPod.DeletionTimestamp == nil {
			if selector.Matches(labels.Set(existingPod.Labels)) {
				score++
			}
		}
	}

	return score, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *ServiceAffinity) NormalizeScore(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	reduceResult := make([]float64, len(scores))
	for _, label := range pl.args.AntiAffinityLabelsPreference {
		if err := pl.updateNodeScoresForLabel(pl.sharedLister, scores, reduceResult, label); err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}
	}

	// Update the result after all labels have been evaluated.
	for i, nodeScore := range reduceResult {
		scores[i].Score = int64(nodeScore)
	}
	return nil
}

// updateNodeScoresForLabel updates the node scores for a single label. Note it does not update the
// original result from the map phase directly, but instead updates the reduceResult, which is used
// to update the original result finally. This makes sure that each call to updateNodeScoresForLabel
// receives the same mapResult to work with.
// Why are doing this? This is a workaround for the migration from priorities to score plugins.
// Historically the priority is designed to handle only one label, and multiple priorities are configured
// to work with multiple labels. Using multiple plugins is not allowed in the new framework. Therefore
// we need to modify the old priority to be able to handle multiple labels so that it can be mapped
// to a single plugin.
// TODO: This will be deprecated soon.
func (pl *ServiceAffinity) updateNodeScoresForLabel(sharedLister schedulerlisters.SharedLister, mapResult framework.NodeScoreList, reduceResult []float64, label string) error {
	var numServicePods int64
	var labelValue string
	podCounts := map[string]int64{}
	labelNodesStatus := map[string]string{}
	maxPriorityFloat64 := float64(framework.MaxNodeScore)

	for _, nodePriority := range mapResult {
		numServicePods += nodePriority.Score
		nodeInfo, err := sharedLister.NodeInfos().Get(nodePriority.Name)
		if err != nil {
			return err
		}
		if !labels.Set(nodeInfo.Node().Labels).Has(label) {
			continue
		}

		labelValue = labels.Set(nodeInfo.Node().Labels).Get(label)
		labelNodesStatus[nodePriority.Name] = labelValue
		podCounts[labelValue] += nodePriority.Score
	}

	//score int - scale of 0-maxPriority
	// 0 being the lowest priority and maxPriority being the highest
	for i, nodePriority := range mapResult {
		labelValue, ok := labelNodesStatus[nodePriority.Name]
		if !ok {
			continue
		}
		// initializing to the default/max node score of maxPriority
		fScore := maxPriorityFloat64
		if numServicePods > 0 {
			fScore = maxPriorityFloat64 * (float64(numServicePods-podCounts[labelValue]) / float64(numServicePods))
		}
		// The score of current label only accounts for 1/len(s.labels) of the total score.
		// The policy API definition only allows a single label to be configured, associated with a weight.
		// This is compensated by the fact that the total weight is the sum of all weights configured
		// in each policy config.
		reduceResult[i] += fScore / float64(len(pl.args.AntiAffinityLabelsPreference))
	}

	return nil
}

// ScoreExtensions of the Score plugin.
func (pl *ServiceAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// addUnsetLabelsToMap backfills missing values with values we find in a map.
func addUnsetLabelsToMap(aL map[string]string, labelsToAdd []string, labelSet labels.Set) {
	for _, l := range labelsToAdd {
		// if the label is already there, dont overwrite it.
		if _, exists := aL[l]; exists {
			continue
		}
		// otherwise, backfill this label.
		if labelSet.Has(l) {
			aL[l] = labelSet.Get(l)
		}
	}
}

// createSelectorFromLabels is used to define a selector that corresponds to the keys in a map.
func createSelectorFromLabels(aL map[string]string) labels.Selector {
	if len(aL) == 0 {
		return labels.Everything()
	}
	return labels.Set(aL).AsSelector()
}

// filterPodsByNamespace filters pods outside a namespace from the given list.
func filterPodsByNamespace(pods []*v1.Pod, ns string) []*v1.Pod {
	filtered := []*v1.Pod{}
	for _, nsPod := range pods {
		if nsPod.Namespace == ns {
			filtered = append(filtered, nsPod)
		}
	}
	return filtered
}

// findLabelsInSet gets as many key/value pairs as possible out of a label set.
func findLabelsInSet(labelsToKeep []string, selector labels.Set) map[string]string {
	aL := make(map[string]string)
	for _, l := range labelsToKeep {
		if selector.Has(l) {
			aL[l] = selector.Get(l)
		}
	}
	return aL
}
