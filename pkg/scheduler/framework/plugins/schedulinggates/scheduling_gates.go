/*
Copyright 2022 The Kubernetes Authors.

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

package schedulinggates

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Name of the plugin used in the plugin registry and configurations.
const Name = names.SchedulingGates

// SchedulingGates checks if a Pod carries .spec.schedulingGates.
type SchedulingGates struct{}

var _ framework.PreEnqueuePlugin = &SchedulingGates{}
var _ framework.EnqueueExtensions = &SchedulingGates{}

func (pl *SchedulingGates) Name() string {
	return Name
}

func (pl *SchedulingGates) PreEnqueue(ctx context.Context, p *v1.Pod) *framework.Status {
	if len(p.Spec.SchedulingGates) == 0 {
		return nil
	}
	gates := make([]string, 0, len(p.Spec.SchedulingGates))
	for _, gate := range p.Spec.SchedulingGates {
		gates = append(gates, gate.Name)
	}
	return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for scheduling gates: %v", gates))
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *SchedulingGates) EventsToRegister() []framework.ClusterEventWithHint {
	return []framework.ClusterEventWithHint{
		{Event: framework.ClusterEvent{Resource: framework.Pod, ActionType: framework.Update}, QueueingHintFn: pl.isSchedulableAfterPodChange},
	}
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, _ framework.Handle, fts feature.Features) (framework.Plugin, error) {
	return &SchedulingGates{}, nil
}

// Pod will become schedulable once it's gates are removed
func (pl *SchedulingGates) isSchedulableAfterPodChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	_, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	if modifiedPod.UID != pod.UID {
		// If the update event is not for targetPod, it wouldn't make targetPod schedulable.
		return framework.QueueSkip, nil
	}

	if len(modifiedPod.Spec.SchedulingGates) == 0 {
		return framework.Queue, nil
	}
	return framework.QueueSkip, nil
}
