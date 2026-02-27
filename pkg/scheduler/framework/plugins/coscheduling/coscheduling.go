/*
Copyright 2025 The Kubernetes Authors.

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

package coscheduling

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha2"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.Coscheduling
)

// Coscheduling is a plugin that facilitates best-effort scheduling for pods
// belonging to a Workload with a Basic scheduling policy.
type Coscheduling struct {
	handle                     fwk.Handle
	podGroupLister             schedulinglisters.PodGroupLister
	enablePodGroupDesiredCount bool
}

var _ fwk.EnqueueExtensions = &Coscheduling{}
var _ fwk.PreEnqueuePlugin = &Coscheduling{}

func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &Coscheduling{
		handle:                     fh,
		podGroupLister:             fh.SharedInformerFactory().Scheduling().V1alpha2().PodGroups().Lister(),
		enablePodGroupDesiredCount: fts.EnablePodGroupDesiredCount,
	}, nil
}

func (pl *Coscheduling) Name() string {
	return Name
}

func (pl *Coscheduling) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		// A new pod being added might be the one that completes a gang, meeting its MinCount requirement.
		// PodSchedulingGroup field is immutable, so there is no need to subscribe on Pod/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add}, QueueingHintFn: helper.IsSchedulableAfterPodAdded},
		// A PodGroup being added can be making a waiting gang schedulable.
		// PodGroups are immutable, so there's no need to handle PodGroup/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.PodGroup, ActionType: fwk.Add}, QueueingHintFn: helper.IsSchedulableAfterPodGroupAdded},
	}, nil
}

func (pl *Coscheduling) PreEnqueue(ctx context.Context, pod *v1.Pod) *fwk.Status {
	if pod.Spec.SchedulingGroup == nil {
		return nil
	}

	namespace := pod.Namespace
	schedulingGroup := pod.Spec.SchedulingGroup

	podGroup, err := pl.podGroupLister.PodGroups(namespace).Get(*schedulingGroup.PodGroupName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// The pod is unschedulable until its PodGroup object is created.
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for pods's pod group %q to appear in scheduling queue", *schedulingGroup.PodGroupName))
		}
		klog.FromContext(ctx).Error(err, "Failed to get pod group", "pod", klog.KObj(pod), "schedulingGroup", schedulingGroup)
		return fwk.AsStatus(fmt.Errorf("failed to get pod group %s/%s", namespace, *schedulingGroup.PodGroupName))
	}

	policy := podGroup.Spec.SchedulingPolicy

	podGroupState, err := pl.handle.PodGroupManager().PodGroupState(namespace, schedulingGroup)
	if err != nil {
		return fwk.AsStatus(err)
	}
	allPods := podGroupState.AllPods()
	var desiredCount *int32
	if policy.Basic != nil {
		desiredCount = policy.Basic.DesiredCount
	} else if policy.Gang != nil {
		desiredCount = policy.Gang.DesiredCount
	}

	if pl.enablePodGroupDesiredCount && desiredCount != nil && len(allPods) < int(*desiredCount) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("introducing delay while all pods count: %d doesn't satisfy desired count requirement: %d", len(allPods), *desiredCount))
	}

	return nil
}
