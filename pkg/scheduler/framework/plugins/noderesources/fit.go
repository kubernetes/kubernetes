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

package noderesources

import (
	"context"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var _ framework.PreFilterPlugin = &Fit{}
var _ framework.FilterPlugin = &Fit{}

const (
	// FitName is the name of the plugin used in the plugin registry and configurations.
	FitName = "NodeResourcesFit"

	// preFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + FitName
)

// Fit is a plugin that checks if a node has sufficient resources.
type Fit struct {
	ignoredResources sets.String
}

// FitArgs holds the args that are used to configure the plugin.
type FitArgs struct {
	// IgnoredResources is the list of resources that NodeResources fit filter
	// should ignore.
	IgnoredResources []string `json:"IgnoredResources,omitempty"`
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	podResourceRequest *nodeinfo.Resource
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	return s
}

// Name returns name of the plugin. It is used in logs, etc.
func (f *Fit) Name() string {
	return FitName
}

// PreFilter invoked at the prefilter extension point.
func (f *Fit) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	s := &preFilterState{
		podResourceRequest: predicates.GetResourceRequest(pod),
	}
	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (f *Fit) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getPodResourceRequest(cycleState *framework.CycleState) (*nodeinfo.Resource, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// The metadata wasn't pre-computed in prefilter. We ignore the error for now since
		// Filter is able to handle that by computing it again.
		klog.V(5).Infof("Error reading %q from cycleState: %v", preFilterStateKey, err)
		return nil, nil
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to NodeResourcesFit.preFilterState error", c)
	}
	return s.podResourceRequest, nil
}

// Filter invoked at the filter extension point.
func (f *Fit) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	r, err := getPodResourceRequest(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	_, reasons, err := predicates.PodFitsResourcesPredicate(pod, r, f.ignoredResources, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// NewFit initializes a new plugin and returns it.
func NewFit(plArgs *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	args := &FitArgs{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}

	fit := &Fit{}
	fit.ignoredResources = sets.NewString(args.IgnoredResources...)
	return fit, nil
}
