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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Name of this plugin.
const Name = "ServiceAffinity"

// Args holds the args that are used to configure the plugin.
type Args struct {
	// Labels should be present for the node to be considered a fit for hosting the pod
	Labels []string `json:"labels,omitempty"`
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
	fitPredicate, predicateMetadataProducer := predicates.NewServiceAffinityPredicate(nodeInfoLister, podLister, serviceLister, args.Labels)

	// Once we generate the predicate we should also Register the Precomputation
	predicates.RegisterPredicateMetadataProducer(predicates.CheckServiceAffinityPred, predicateMetadataProducer)

	return &ServiceAffinity{
		predicate: fitPredicate,
	}, nil
}

// ServiceAffinity is a plugin that checks service affinity.
type ServiceAffinity struct {
	predicate predicates.FitPredicate
}

var _ framework.FilterPlugin = &ServiceAffinity{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *ServiceAffinity) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *ServiceAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, ok := migration.PredicateMetadata(cycleState).(predicates.PredicateMetadata)
	if !ok {
		return framework.NewStatus(framework.Error, "looking up PredicateMetadata")
	}
	_, reasons, err := pl.predicate(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}
