/*
Copyright 2023 The Kubernetes Authors.

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

package runtime

import (
	"context"

	v1 "k8s.io/api/core/v1"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

type instrumentedFilterPlugin struct {
	framework.FilterPlugin

	metric compbasemetrics.CounterMetric
}

var _ framework.FilterPlugin = &instrumentedFilterPlugin{}

func (p *instrumentedFilterPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	p.metric.Inc()
	return p.FilterPlugin.Filter(ctx, state, pod, nodeInfo)
}

type instrumentedPreFilterPlugin struct {
	framework.PreFilterPlugin

	metric compbasemetrics.CounterMetric
}

var _ framework.PreFilterPlugin = &instrumentedPreFilterPlugin{}

func (p *instrumentedPreFilterPlugin) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	result, status := p.PreFilterPlugin.PreFilter(ctx, state, pod)
	if !status.IsSkip() {
		p.metric.Inc()
	}
	return result, status
}
