/*
Copyright 2020 The Kubernetes Authors.

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

package framework

import (
	"context"
	"fmt"
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// FitPredicate is a function type which is used in fake extender.
type FitPredicate func(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status

// PriorityFunc is a function type which is used in fake extender.
type PriorityFunc func(pod *v1.Pod, nodes []*framework.NodeInfo) (*framework.NodeScoreList, error)

// PriorityConfig is used in fake extender to perform Prioritize function.
type PriorityConfig struct {
	Function PriorityFunc
	Weight   int64
}

// ErrorPredicateExtender implements FitPredicate function to always return error status.
func ErrorPredicateExtender(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Error, "some error")
}

// FalsePredicateExtender implements FitPredicate function to always return unschedulable status.
func FalsePredicateExtender(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("pod is unschedulable on the node %q", node.Node().Name))
}

// TruePredicateExtender implements FitPredicate function to always return success status.
func TruePredicateExtender(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Success)
}

// Node1PredicateExtender implements FitPredicate function to return true
// when the given node's name is "node1"; otherwise return false.
func Node1PredicateExtender(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	if node.Node().Name == "node1" {
		return fwk.NewStatus(fwk.Success)
	}
	return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Node().Name))
}

// Node2PredicateExtender implements FitPredicate function to return true
// when the given node's name is "node2"; otherwise return false.
func Node2PredicateExtender(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	if node.Node().Name == "node2" {
		return fwk.NewStatus(fwk.Success)
	}
	return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Node().Name))
}

// ErrorPrioritizerExtender implements PriorityFunc function to always return error.
func ErrorPrioritizerExtender(pod *v1.Pod, nodes []*framework.NodeInfo) (*framework.NodeScoreList, error) {
	return &framework.NodeScoreList{}, fmt.Errorf("some error")
}

// Node1PrioritizerExtender implements PriorityFunc function to give score 10
// if the given node's name is "node1"; otherwise score 1.
func Node1PrioritizerExtender(pod *v1.Pod, nodes []*framework.NodeInfo) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Node().Name == "node1" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Node().Name, Score: int64(score)})
	}
	return &result, nil
}

// Node2PrioritizerExtender implements PriorityFunc function to give score 10
// if the given node's name is "node2"; otherwise score 1.
func Node2PrioritizerExtender(pod *v1.Pod, nodes []*framework.NodeInfo) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Node().Name == "node2" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Node().Name, Score: int64(score)})
	}
	return &result, nil
}

type node2PrioritizerPlugin struct{}

// NewNode2PrioritizerPlugin returns a factory function to build node2PrioritizerPlugin.
func NewNode2PrioritizerPlugin() frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &node2PrioritizerPlugin{}, nil
	}
}

// Name returns name of the plugin.
func (pl *node2PrioritizerPlugin) Name() string {
	return "Node2Prioritizer"
}

// Score return score 100 if the given nodeName is "node2"; otherwise return score 10.
func (pl *node2PrioritizerPlugin) Score(_ context.Context, _ fwk.CycleState, _ *v1.Pod, nodeInfo *framework.NodeInfo) (int64, *fwk.Status) {
	score := 10
	if nodeInfo.Node().Name == "node2" {
		score = 100
	}
	return int64(score), nil
}

// ScoreExtensions returns nil.
func (pl *node2PrioritizerPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// FakeExtender is a data struct which implements the Extender interface.
type FakeExtender struct {
	// ExtenderName indicates this fake extender's name.
	// Note that extender name should be unique.
	ExtenderName     string
	Predicates       []FitPredicate
	Prioritizers     []PriorityConfig
	Weight           int64
	NodeCacheCapable bool
	FilteredNodes    []*framework.NodeInfo
	UnInterested     bool
	Ignorable        bool
	Binder           func() error

	// Cached node information for fake extender
	CachedNodeNameToInfo map[string]*framework.NodeInfo
}

const defaultFakeExtenderName = "defaultFakeExtender"

// Name returns name of the extender.
func (f *FakeExtender) Name() string {
	if f.ExtenderName == "" {
		// If ExtenderName is unset, use default name.
		return defaultFakeExtenderName
	}
	return f.ExtenderName
}

// IsIgnorable returns a bool value indicating whether internal errors can be ignored.
func (f *FakeExtender) IsIgnorable() bool {
	return f.Ignorable
}

// SupportsPreemption returns true indicating the extender supports preemption.
func (f *FakeExtender) SupportsPreemption() bool {
	// Assume preempt verb is always defined.
	return true
}

// ProcessPreemption implements the extender preempt function.
func (f *FakeExtender) ProcessPreemption(
	pod *v1.Pod,
	nodeNameToVictims map[string]*extenderv1.Victims,
	nodeInfos framework.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	nodeNameToVictimsCopy := map[string]*extenderv1.Victims{}
	// We don't want to change the original nodeNameToVictims
	for k, v := range nodeNameToVictims {
		// In real world implementation, extender's user should have their own way to get node object
		// by name if needed (e.g. query kube-apiserver etc).
		//
		// For test purpose, we just use node from parameters directly.
		nodeNameToVictimsCopy[k] = v
	}

	// If Extender.ProcessPreemption ever gets extended with a context parameter, then the logger should be retrieved from that.
	// Now, in order not to modify the Extender interface, we get the logger from klog.TODO()
	logger := klog.TODO()
	for nodeName, victims := range nodeNameToVictimsCopy {
		// Try to do preemption on extender side.
		nodeInfo, _ := nodeInfos.Get(nodeName)
		extenderVictimPods, extenderPDBViolations, fits, err := f.selectVictimsOnNodeByExtender(logger, pod, nodeInfo)
		if err != nil {
			return nil, err
		}
		// If it's unfit after extender's preemption, this node is unresolvable by preemption overall,
		// let's remove it from potential preemption nodes.
		if !fits {
			delete(nodeNameToVictimsCopy, nodeName)
		} else {
			// Append new victims to original victims
			nodeNameToVictimsCopy[nodeName].Pods = append(victims.Pods, extenderVictimPods...)
			nodeNameToVictimsCopy[nodeName].NumPDBViolations = victims.NumPDBViolations + int64(extenderPDBViolations)
		}
	}
	return nodeNameToVictimsCopy, nil
}

// selectVictimsOnNodeByExtender checks the given nodes->pods map with predicates on extender's side.
// Returns:
// 1. More victim pods (if any) amended by preemption phase of extender.
// 2. Number of violating victim (used to calculate PDB).
// 3. Fits or not after preemption phase on extender's side.
func (f *FakeExtender) selectVictimsOnNodeByExtender(logger klog.Logger, pod *v1.Pod, node *framework.NodeInfo) ([]*v1.Pod, int, bool, error) {
	// If a extender support preemption but have no cached node info, let's run filter to make sure
	// default scheduler's decision still stand with given pod and node.
	if !f.NodeCacheCapable {
		err := f.runPredicate(pod, node)
		if err.IsSuccess() {
			return []*v1.Pod{}, 0, true, nil
		} else if err.IsRejected() {
			return nil, 0, false, nil
		} else {
			return nil, 0, false, err.AsError()
		}
	}

	// Otherwise, as a extender support preemption and have cached node info, we will assume cachedNodeNameToInfo is available
	// and get cached node info by given node name.
	nodeInfoCopy := f.CachedNodeNameToInfo[node.Node().Name].Snapshot()

	var potentialVictims []*v1.Pod

	removePod := func(rp *v1.Pod) error {
		return nodeInfoCopy.RemovePod(logger, rp)
	}
	addPod := func(ap *v1.Pod) {
		nodeInfoCopy.AddPod(ap)
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := corev1helpers.PodPriority(pod)
	for _, p := range nodeInfoCopy.Pods {
		if corev1helpers.PodPriority(p.Pod) < podPriority {
			potentialVictims = append(potentialVictims, p.Pod)
			if err := removePod(p.Pod); err != nil {
				return nil, 0, false, err
			}
		}
	}
	sort.Slice(potentialVictims, func(i, j int) bool { return util.MoreImportantPod(potentialVictims[i], potentialVictims[j]) })

	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption.
	status := f.runPredicate(pod, nodeInfoCopy)
	if status.IsSuccess() {
		// pass
	} else if status.IsRejected() {
		// does not fit
		return nil, 0, false, nil
	} else {
		// internal errors
		return nil, 0, false, status.AsError()
	}

	var victims []*v1.Pod

	// TODO(harry): handle PDBs in the future.
	numViolatingVictim := 0

	reprievePod := func(p *v1.Pod) bool {
		addPod(p)
		status := f.runPredicate(pod, nodeInfoCopy)
		if !status.IsSuccess() {
			if err := removePod(p); err != nil {
				return false
			}
			victims = append(victims, p)
		}
		return status.IsSuccess()
	}

	// For now, assume all potential victims to be non-violating.
	// Now we try to reprieve non-violating victims.
	for _, p := range potentialVictims {
		reprievePod(p)
	}

	return victims, numViolatingVictim, true, nil
}

// runPredicate run predicates of extender one by one for given pod and node.
// Returns: fits or not.
func (f *FakeExtender) runPredicate(pod *v1.Pod, node *framework.NodeInfo) *fwk.Status {
	for _, predicate := range f.Predicates {
		status := predicate(pod, node)
		if !status.IsSuccess() {
			return status
		}
	}
	return fwk.NewStatus(fwk.Success)
}

// Filter implements the extender Filter function.
func (f *FakeExtender) Filter(pod *v1.Pod, nodes []*framework.NodeInfo) ([]*framework.NodeInfo, extenderv1.FailedNodesMap, extenderv1.FailedNodesMap, error) {
	var filtered []*framework.NodeInfo
	failedNodesMap := extenderv1.FailedNodesMap{}
	failedAndUnresolvableMap := extenderv1.FailedNodesMap{}
	for _, node := range nodes {
		status := f.runPredicate(pod, node)
		if status.IsSuccess() {
			filtered = append(filtered, node)
		} else if status.Code() == fwk.Unschedulable {
			failedNodesMap[node.Node().Name] = fmt.Sprintf("FakeExtender: node %q failed", node.Node().Name)
		} else if status.Code() == fwk.UnschedulableAndUnresolvable {
			failedAndUnresolvableMap[node.Node().Name] = fmt.Sprintf("FakeExtender: node %q failed and unresolvable", node.Node().Name)
		} else {
			return nil, nil, nil, status.AsError()
		}
	}

	f.FilteredNodes = filtered
	if f.NodeCacheCapable {
		return filtered, failedNodesMap, failedAndUnresolvableMap, nil
	}
	return filtered, failedNodesMap, failedAndUnresolvableMap, nil
}

// Prioritize implements the extender Prioritize function.
func (f *FakeExtender) Prioritize(pod *v1.Pod, nodes []*framework.NodeInfo) (*extenderv1.HostPriorityList, int64, error) {
	result := extenderv1.HostPriorityList{}
	combinedScores := map[string]int64{}
	for _, prioritizer := range f.Prioritizers {
		weight := prioritizer.Weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.Function
		prioritizedList, err := priorityFunc(pod, nodes)
		if err != nil {
			return &extenderv1.HostPriorityList{}, 0, err
		}
		for _, hostEntry := range *prioritizedList {
			combinedScores[hostEntry.Name] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		result = append(result, extenderv1.HostPriority{Host: host, Score: score})
	}
	return &result, f.Weight, nil
}

// Bind implements the extender Bind function.
func (f *FakeExtender) Bind(binding *v1.Binding) error {
	if f.Binder != nil {
		return f.Binder()
	}
	if len(f.FilteredNodes) != 0 {
		for _, node := range f.FilteredNodes {
			if node.Node().Name == binding.Target.Name {
				f.FilteredNodes = nil
				return nil
			}
		}
		err := fmt.Errorf("Node %v not in filtered nodes %v", binding.Target.Name, f.FilteredNodes)
		f.FilteredNodes = nil
		return err
	}
	return nil
}

// IsBinder returns true indicating the extender implements the Binder function.
func (f *FakeExtender) IsBinder() bool {
	return true
}

// IsPrioritizer returns true if there are any prioritizers.
func (f *FakeExtender) IsPrioritizer() bool {
	return len(f.Prioritizers) > 0
}

// IsFilter returns true if there are any filters.
func (f *FakeExtender) IsFilter() bool {
	return len(f.Predicates) > 0
}

// IsInterested returns a bool indicating whether this extender is interested in this Pod.
func (f *FakeExtender) IsInterested(pod *v1.Pod) bool {
	return !f.UnInterested
}

var _ framework.Extender = &FakeExtender{}
