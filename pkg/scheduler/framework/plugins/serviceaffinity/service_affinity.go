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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "ServiceAffinity"

// Args holds the args that are used to configure the plugin.
type Args struct {
	// Labels are homogeneous for pods that are scheduled to a node.
	// (i.e. it returns true IFF this pod can be added to this node such that all other pods in
	// the same service are running on nodes with the exact same values for Labels).
	AffinityLabels []string `json:"labels,omitempty"`
	// AntiAffinityLabelsPreference are the labels to consider for service anti affinity scoring.
	AntiAffinityLabelsPreference []string `json:"antiAffinityLabelsPreference,omitempty"`
}

// New initializes a new plugin and returns it.
func New(plArgs *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	args := &Args{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}
	informerFactory := handle.SharedInformerFactory()
	nodeInfoLister := handle.SnapshotSharedLister().NodeInfos()
	podLister := handle.SnapshotSharedLister().Pods()
	serviceLister := informerFactory.Core().V1().Services().Lister()

	fitPredicate, predicateMetadataProducer := predicates.NewServiceAffinityPredicate(nodeInfoLister, podLister, serviceLister, args.AffinityLabels)
	// Once we generate the predicate we should also Register the Precomputation
	predicates.RegisterPredicateMetadataProducer(predicates.CheckServiceAffinityPred, predicateMetadataProducer)

	priorityMapFunction, priorityReduceFunction := priorities.NewServiceAntiAffinityPriority(podLister, serviceLister, args.AntiAffinityLabelsPreference)

	return &ServiceAffinity{
		handle:                 handle,
		predicate:              fitPredicate,
		priorityMapFunction:    priorityMapFunction,
		priorityReduceFunction: priorityReduceFunction,
	}, nil
}

// ServiceAffinity is a plugin that checks service affinity.
type ServiceAffinity struct {
	handle                 framework.FrameworkHandle
	predicate              predicates.FitPredicate
	priorityMapFunction    priorities.PriorityMapFunction
	priorityReduceFunction priorities.PriorityReduceFunction
}

var _ framework.FilterPlugin = &ServiceAffinity{}
var _ framework.ScorePlugin = &ServiceAffinity{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *ServiceAffinity) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *ServiceAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, ok := migration.PredicateMetadata(cycleState).(predicates.Metadata)
	if !ok {
		return framework.NewStatus(framework.Error, "looking up Metadata")
	}
	_, reasons, err := pl.predicate(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the Score extension point.
func (pl *ServiceAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}
	meta := migration.PriorityMetadata(state)
	s, err := pl.priorityMapFunction(pod, meta, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// NormalizeScore invoked after scoring all nodes.
func (pl *ServiceAffinity) NormalizeScore(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	// Note that priorityReduceFunction doesn't use priority metadata, hence passing nil here.
	err := pl.priorityReduceFunction(pod, nil, pl.handle.SnapshotSharedLister(), scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *ServiceAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}
