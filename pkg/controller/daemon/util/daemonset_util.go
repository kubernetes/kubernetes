/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/scheme"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

// CreatePodTemplate returns copy of provided template with additional
// label which contains templateGeneration (for backward compatibility),
// hash of provided template and sets default daemon tolerations.
func CreatePodTemplate(template v1.PodTemplateSpec, generation int64, hash string) v1.PodTemplateSpec {
	obj, _ := scheme.Scheme.DeepCopy(template)
	newTemplate := obj.(v1.PodTemplateSpec)
	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint notReady:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns not ready.
	v1helper.AddOrUpdateTolerationInPodSpec(&newTemplate.Spec, &v1.Toleration{
		Key:      algorithm.TaintNodeNotReady,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint unreachable:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns unreachable.
	v1helper.AddOrUpdateTolerationInPodSpec(&newTemplate.Spec, &v1.Toleration{
		Key:      algorithm.TaintNodeUnreachable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	// According to TaintNodesByCondition feature, all DaemonSet pods should tolerate
	// MemoryPressure and DisPressure taints, and the critical pods should tolerate
	// OutOfDisk taint.
	v1helper.AddOrUpdateTolerationInPodSpec(&newTemplate.Spec, &v1.Toleration{
		Key:      algorithm.TaintNodeDiskPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	v1helper.AddOrUpdateTolerationInPodSpec(&newTemplate.Spec, &v1.Toleration{
		Key:      algorithm.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	if utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation) &&
		kubelettypes.IsCritical(newTemplate.Namespace, newTemplate.Annotations) {
		v1helper.AddOrUpdateTolerationInPodSpec(&newTemplate.Spec, &v1.Toleration{
			Key:      algorithm.TaintNodeOutOfDisk,
			Operator: v1.TolerationOpExists,
			Effect:   v1.TaintEffectNoExecute,
		})
	}

	templateGenerationStr := fmt.Sprint(generation)
	newTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		template.ObjectMeta.Labels,
		extensions.DaemonSetTemplateGenerationKey,
		templateGenerationStr,
	)
	// TODO: do we need to validate if the DaemonSet is RollingUpdate or not?
	if len(hash) > 0 {
		newTemplate.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey] = hash
	}
	return newTemplate
}

// IsPodUpdate checks if pod contains label value that either matches templateGeneration or hash
func IsPodUpdated(dsTemplateGeneration int64, pod *v1.Pod, hash string) bool {
	// Compare with hash to see if the pod is updated, need to maintain backward compatibility of templateGeneration
	templateMatches := pod.Labels[extensions.DaemonSetTemplateGenerationKey] == fmt.Sprint(dsTemplateGeneration)
	hashMatches := len(hash) > 0 && pod.Labels[extensions.DefaultDaemonSetUniqueLabelKey] == hash
	return hashMatches || templateMatches
}

// SplitByAvailablePods splits provided daemon set pods by availabilty
func SplitByAvailablePods(minReadySeconds int32, pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod) {
	unavailablePods := []*v1.Pod{}
	availablePods := []*v1.Pod{}
	for _, pod := range pods {
		if podutil.IsPodAvailable(pod, minReadySeconds, metav1.Now()) {
			availablePods = append(availablePods, pod)
		} else {
			unavailablePods = append(unavailablePods, pod)
		}
	}
	return availablePods, unavailablePods
}
