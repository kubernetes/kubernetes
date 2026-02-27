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

package helper

import (
	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// matchingSchedulingGroup returns true if two pods belong to the same scheduling group.
func matchingSchedulingGroup(pod1, pod2 *v1.Pod) bool {
	return pod1.Namespace == pod2.Namespace &&
		pod1.Spec.SchedulingGroup != nil &&
		pod2.Spec.SchedulingGroup != nil &&
		*pod1.Spec.SchedulingGroup.PodGroupName == *pod2.Spec.SchedulingGroup.PodGroupName
}

// IsSchedulableAfterPodAdded is a queueing hint function to evaluate if a newly added pod makes the target pod schedulable.
func IsSchedulableAfterPodAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if !matchingSchedulingGroup(pod, addedPod) {
		logger.V(5).Info("another pod was added but it doesn't match the target pod's scheduling group",
			"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPod", klog.KObj(addedPod), "addedPodSchedulingGroup", addedPod.Spec.SchedulingGroup)
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("another pod was added and it matches the target pod's scheduling group, which may make the pod schedulable",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPod", klog.KObj(addedPod), "addedPodSchedulingGroup", addedPod.Spec.SchedulingGroup)
	return fwk.Queue, nil
}

// IsSchedulableAfterPodGroupAdded is a queueing hint function to evaluate if a newly added workload makes the target pod schedulable.
func IsSchedulableAfterPodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPodGroup, err := util.As[*schedulingapi.PodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup == nil || pod.Namespace != addedPodGroup.Namespace || *pod.Spec.SchedulingGroup.PodGroupName != addedPodGroup.Name {
		logger.V(5).Info("pod group was added but it doesn't match the target pod's scheduling group",
			"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPodGroup", klog.KObj(addedPodGroup))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("pod group was added and it matches the target pod's scheduling group, which may make the pod schedulable",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPodGroup", klog.KObj(addedPodGroup))
	return fwk.Queue, nil
}
