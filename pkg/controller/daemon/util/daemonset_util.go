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
	"strconv"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	intstrutil "k8s.io/apimachinery/pkg/util/intstr"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

// GetTemplateGeneration gets the template generation associated with a v1.DaemonSet by extracting it from the
// deprecated annotation. If no annotation is found nil is returned. If the annotation is found and fails to parse
// nil is returned with an error. If the generation can be parsed from the annotation, a pointer to the parsed int64
// value is returned.
func GetTemplateGeneration(ds *apps.DaemonSet) (*int64, error) {
	annotation, found := ds.Annotations[apps.DeprecatedTemplateGeneration]
	if !found {
		return nil, nil
	}
	generation, err := strconv.ParseInt(annotation, 10, 64)
	if err != nil {
		return nil, err
	}
	return &generation, nil
}

// AddOrUpdateDaemonPodTolerations apply necessary tolerations to DaemonSet Pods, e.g. node.kubernetes.io/not-ready:NoExecute.
func AddOrUpdateDaemonPodTolerations(spec *v1.PodSpec) {
	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint notReady:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns not ready.
	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodeNotReady,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint unreachable:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns unreachable.
	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodeUnreachable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	// According to TaintNodesByCondition feature, all DaemonSet pods should tolerate
	// MemoryPressure, DiskPressure, PIDPressure, Unschedulable and NetworkUnavailable taints.
	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodeDiskPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodePIDPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
		Key:      v1.TaintNodeUnschedulable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	})

	if spec.HostNetwork {
		v1helper.AddOrUpdateTolerationInPodSpec(spec, &v1.Toleration{
			Key:      v1.TaintNodeNetworkUnavailable,
			Operator: v1.TolerationOpExists,
			Effect:   v1.TaintEffectNoSchedule,
		})
	}
}

// CreatePodTemplate returns copy of provided template with additional
// label which contains templateGeneration (for backward compatibility),
// hash of provided template and sets default daemon tolerations.
func CreatePodTemplate(template v1.PodTemplateSpec, generation *int64, hash string) v1.PodTemplateSpec {
	newTemplate := *template.DeepCopy()

	AddOrUpdateDaemonPodTolerations(&newTemplate.Spec)

	if newTemplate.ObjectMeta.Labels == nil {
		newTemplate.ObjectMeta.Labels = make(map[string]string)
	}
	if generation != nil {
		newTemplate.ObjectMeta.Labels[extensions.DaemonSetTemplateGenerationKey] = fmt.Sprint(*generation)
	}
	// TODO: do we need to validate if the DaemonSet is RollingUpdate or not?
	if len(hash) > 0 {
		newTemplate.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey] = hash
	}
	return newTemplate
}

// AllowsSurge returns true if the daemonset allows more than a single pod on any node.
func AllowsSurge(ds *apps.DaemonSet) bool {
	maxSurge, err := SurgeCount(ds, 1)
	return err == nil && maxSurge > 0
}

// SurgeCount returns 0 if surge is not requested, the expected surge number to allow
// out of numberToSchedule if surge is configured, or an error if the surge percentage
// requested is invalid.
func SurgeCount(ds *apps.DaemonSet, numberToSchedule int) (int, error) {
	if ds.Spec.UpdateStrategy.Type != apps.RollingUpdateDaemonSetStrategyType {
		return 0, nil
	}

	r := ds.Spec.UpdateStrategy.RollingUpdate
	if r == nil {
		return 0, nil
	}
	// If surge is not requested, we should default to 0.
	if r.MaxSurge == nil {
		return 0, nil
	}
	return intstrutil.GetScaledValueFromIntOrPercent(r.MaxSurge, numberToSchedule, true)
}

// UnavailableCount returns 0 if unavailability is not requested, the expected
// unavailability number to allow out of numberToSchedule if requested, or an error if
// the unavailability percentage requested is invalid.
func UnavailableCount(ds *apps.DaemonSet, numberToSchedule int) (int, error) {
	if ds.Spec.UpdateStrategy.Type != apps.RollingUpdateDaemonSetStrategyType {
		return 0, nil
	}
	r := ds.Spec.UpdateStrategy.RollingUpdate
	if r == nil {
		return 0, nil
	}
	return intstrutil.GetScaledValueFromIntOrPercent(r.MaxUnavailable, numberToSchedule, true)
}

// IsPodUpdated checks if pod contains label value that either matches templateGeneration or hash
func IsPodUpdated(pod *v1.Pod, hash string, dsTemplateGeneration *int64) bool {
	// Compare with hash to see if the pod is updated, need to maintain backward compatibility of templateGeneration
	templateMatches := dsTemplateGeneration != nil &&
		pod.Labels[extensions.DaemonSetTemplateGenerationKey] == fmt.Sprint(*dsTemplateGeneration)
	hashMatches := len(hash) > 0 && pod.Labels[extensions.DefaultDaemonSetUniqueLabelKey] == hash
	return hashMatches || templateMatches
}

// ReplaceDaemonSetPodNodeNameNodeAffinity replaces the RequiredDuringSchedulingIgnoredDuringExecution
// NodeAffinity of the given affinity with a new NodeAffinity that selects the given nodeName.
// Note that this function assumes that no NodeAffinity conflicts with the selected nodeName.
func ReplaceDaemonSetPodNodeNameNodeAffinity(affinity *v1.Affinity, nodename string) *v1.Affinity {
	nodeSelReq := v1.NodeSelectorRequirement{
		Key:      metav1.ObjectNameField,
		Operator: v1.NodeSelectorOpIn,
		Values:   []string{nodename},
	}

	nodeSelector := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchFields: []v1.NodeSelectorRequirement{nodeSelReq},
			},
		},
	}

	if affinity == nil {
		return &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector,
			},
		}
	}

	if affinity.NodeAffinity == nil {
		affinity.NodeAffinity = &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector,
		}
		return affinity
	}

	nodeAffinity := affinity.NodeAffinity

	if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = nodeSelector
		return affinity
	}

	// Replace node selector with the new one.
	nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = []v1.NodeSelectorTerm{
		{
			MatchFields: []v1.NodeSelectorRequirement{nodeSelReq},
		},
	}

	return affinity
}

// GetTargetNodeName get the target node name of DaemonSet pods. If `.spec.NodeName` is not empty (nil),
// return `.spec.NodeName`; otherwise, retrieve node name of pending pods from NodeAffinity. Return error
// if failed to retrieve node name from `.spec.NodeName` and NodeAffinity.
func GetTargetNodeName(pod *v1.Pod) (string, error) {
	if len(pod.Spec.NodeName) != 0 {
		return pod.Spec.NodeName, nil
	}

	// Retrieve node name of unscheduled pods from NodeAffinity
	if pod.Spec.Affinity == nil ||
		pod.Spec.Affinity.NodeAffinity == nil ||
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return "", fmt.Errorf("no spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution for pod %s/%s",
			pod.Namespace, pod.Name)
	}

	terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	if len(terms) < 1 {
		return "", fmt.Errorf("no nodeSelectorTerms in requiredDuringSchedulingIgnoredDuringExecution of pod %s/%s",
			pod.Namespace, pod.Name)
	}

	for _, term := range terms {
		for _, exp := range term.MatchFields {
			if exp.Key == metav1.ObjectNameField &&
				exp.Operator == v1.NodeSelectorOpIn {
				if len(exp.Values) != 1 {
					return "", fmt.Errorf("the matchFields value of '%s' is not unique for pod %s/%s",
						metav1.ObjectNameField, pod.Namespace, pod.Name)
				}

				return exp.Values[0], nil
			}
		}
	}

	return "", fmt.Errorf("no node name found for pod %s/%s", pod.Namespace, pod.Name)
}
