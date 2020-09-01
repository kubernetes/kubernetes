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

package testing

import (
	"context"
	"fmt"
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// FitPredicate is a function type which is used in fake extender.
type FitPredicate func(pod *v1.Pod, node *v1.Node) (bool, error)

// PriorityFunc is a function type which is used in fake extender.
type PriorityFunc func(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error)

// PriorityConfig is used in fake extender to perform Prioritize function.
type PriorityConfig struct {
	Function PriorityFunc
	Weight   int64
}

// ErrorPredicateExtender implements FitPredicate function to always return error.
func ErrorPredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return false, fmt.Errorf("some error")
}

// FalsePredicateExtender implements FitPredicate function to always return false.
func FalsePredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return false, nil
}

// TruePredicateExtender implements FitPredicate function to always return true.
func TruePredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return true, nil
}

// Node1PredicateExtender implements FitPredicate function to return true
// when the given node's name is "node1"; otherwise return false.
func Node1PredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "node1" {
		return true, nil
	}
	return false, nil
}

// Node2PredicateExtender implements FitPredicate function to return true
// when the given node's name is "node2"; otherwise return false.
func Node2PredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "node2" {
		return true, nil
	}
	return false, nil
}

// ErrorPrioritizerExtender implements PriorityFunc function to always return error.
func ErrorPrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	return &framework.NodeScoreList{}, fmt.Errorf("some error")
}

// Node1PrioritizerExtender implements PriorityFunc function to give score 10
// if the given node's name is "node1"; otherwise score 1.
func Node1PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "node1" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Name, Score: int64(score)})
	}
	return &result, nil
}

// Node2PrioritizerExtender implements PriorityFunc function to give score 10
// if the given node's name is "node2"; otherwise score 1.
func Node2PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "node2" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Name, Score: int64(score)})
	}
	return &result, nil
}

type node2PrioritizerPlugin struct{}

// NewNode2PrioritizerPlugin returns a factory function to build node2PrioritizerPlugin.
func NewNode2PrioritizerPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &node2PrioritizerPlugin{}, nil
	}
}

// Name returns name of the plugin.
func (pl *node2PrioritizerPlugin) Name() string {
	return "Node2Prioritizer"
}

// Score return score 100 if the given nodeName is "node2"; otherwise return score 10.
func (pl *node2PrioritizerPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score := 10
	if nodeName == "node2" {
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
	Predicates       []FitPredicate
	Prioritizers     []PriorityConfig
	Weight           int64
	NodeCacheCapable bool
	FilteredNodes    []*v1.Node
	UnInterested     bool
	Ignorable        bool

	// Cached node information for fake extender
	CachedNodeNameToInfo map[string]*framework.NodeInfo
}

// Name returns name of the extender.
func (f *FakeExtender) Name() string {
	return "FakeExtender"
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

	for nodeName, victims := range nodeNameToVictimsCopy {
		// Try to do preemption on extender side.
		nodeInfo, _ := nodeInfos.Get(nodeName)
		extenderVictimPods, extenderPDBViolations, fits, err := f.selectVictimsOnNodeByExtender(pod, nodeInfo.Node())
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
func (f *FakeExtender) selectVictimsOnNodeByExtender(pod *v1.Pod, node *v1.Node) ([]*v1.Pod, int, bool, error) {
	// If a extender support preemption but have no cached node info, let's run filter to make sure
	// default scheduler's decision still stand with given pod and node.
	if !f.NodeCacheCapable {
		fits, err := f.runPredicate(pod, node)
		if err != nil {
			return nil, 0, false, err
		}
		if !fits {
			return nil, 0, false, nil
		}
		return []*v1.Pod{}, 0, true, nil
	}

	// Otherwise, as a extender support preemption and have cached node info, we will assume cachedNodeNameToInfo is available
	// and get cached node info by given node name.
	nodeInfoCopy := f.CachedNodeNameToInfo[node.GetName()].Clone()

	var potentialVictims []*v1.Pod

	removePod := func(rp *v1.Pod) {
		nodeInfoCopy.RemovePod(rp)
	}
	addPod := func(ap *v1.Pod) {
		nodeInfoCopy.AddPod(ap)
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := podutil.GetPodPriority(pod)
	for _, p := range nodeInfoCopy.Pods {
		if podutil.GetPodPriority(p.Pod) < podPriority {
			potentialVictims = append(potentialVictims, p.Pod)
			removePod(p.Pod)
		}
	}
	sort.Slice(potentialVictims, func(i, j int) bool { return util.MoreImportantPod(potentialVictims[i], potentialVictims[j]) })

	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption.
	fits, err := f.runPredicate(pod, nodeInfoCopy.Node())
	if err != nil {
		return nil, 0, false, err
	}
	if !fits {
		return nil, 0, false, nil
	}

	var victims []*v1.Pod

	// TODO(harry): handle PDBs in the future.
	numViolatingVictim := 0

	reprievePod := func(p *v1.Pod) bool {
		addPod(p)
		fits, _ := f.runPredicate(pod, nodeInfoCopy.Node())
		if !fits {
			removePod(p)
			victims = append(victims, p)
		}
		return fits
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
func (f *FakeExtender) runPredicate(pod *v1.Pod, node *v1.Node) (bool, error) {
	fits := true
	var err error
	for _, predicate := range f.Predicates {
		fits, err = predicate(pod, node)
		if err != nil {
			return false, err
		}
		if !fits {
			break
		}
	}
	return fits, nil
}

// Filter implements the extender Filter function.
func (f *FakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) ([]*v1.Node, extenderv1.FailedNodesMap, error) {
	var filtered []*v1.Node
	failedNodesMap := extenderv1.FailedNodesMap{}
	for _, node := range nodes {
		fits, err := f.runPredicate(pod, node)
		if err != nil {
			return []*v1.Node{}, extenderv1.FailedNodesMap{}, err
		}
		if fits {
			filtered = append(filtered, node)
		} else {
			failedNodesMap[node.Name] = "FakeExtender failed"
		}
	}

	f.FilteredNodes = filtered
	if f.NodeCacheCapable {
		return filtered, failedNodesMap, nil
	}
	return filtered, failedNodesMap, nil
}

// Prioritize implements the extender Prioritize function.
func (f *FakeExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*extenderv1.HostPriorityList, int64, error) {
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
	if len(f.FilteredNodes) != 0 {
		for _, node := range f.FilteredNodes {
			if node.Name == binding.Target.Name {
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

// IsInterested returns a bool true indicating whether extender
func (f *FakeExtender) IsInterested(pod *v1.Pod) bool {
	return !f.UnInterested
}

var _ framework.Extender = &FakeExtender{}
