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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// PodTopologySpread is a plugin that ensures pod's topologySpreadConstraints is satisfied.
type PodTopologySpread struct{}

var _ framework.FilterPlugin = &PodTopologySpread{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "PodTopologySpread"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodTopologySpread) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *PodTopologySpread) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, ok := migration.PredicateMetadata(cycleState).(predicates.PredicateMetadata)
	if !ok {
		return migration.ErrorToFrameworkStatus(fmt.Errorf("%+v convert to predicates.PredicateMetadata error", cycleState))
	}
	_, reasons, err := predicates.EvenPodsSpreadPredicate(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &PodTopologySpread{}, nil
}
