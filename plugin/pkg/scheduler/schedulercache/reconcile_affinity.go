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

package schedulercache

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/features"
)

// Reconcile api and annotation affinity definitions.
// When alpha affinity feature is not enabled, always take affinity
// from PodSpec.When alpha affinity feature is enabled, if affinity
// is not set in PodSpec, take affinity from annotation.
// When alpha affinity feature is enabled, if affinity is set in PodSpec,
// take node affinity, pod affinity, and pod anti-affinity individually
// using the following rule: take affinity from PodSpec if it is defined,
// otherwise take from annotation if it is defined.
// TODO: remove when alpha support for affinity is removed
func ReconcileAffinity(pod *v1.Pod) *v1.Affinity {
	affinity := pod.Spec.Affinity
	if utilfeature.DefaultFeatureGate.Enabled(features.AffinityInAnnotations) {
		annotationsAffinity, _ := v1helper.GetAffinityFromPodAnnotations(pod.Annotations)
		if affinity == nil && annotationsAffinity != nil {
			affinity = annotationsAffinity
		} else if annotationsAffinity != nil {
			if affinity != nil && affinity.NodeAffinity == nil && annotationsAffinity.NodeAffinity != nil {
				affinity.NodeAffinity = annotationsAffinity.NodeAffinity
			}
			if affinity != nil && affinity.PodAffinity == nil && annotationsAffinity.PodAffinity != nil {
				affinity.PodAffinity = annotationsAffinity.PodAffinity
			}
			if affinity != nil && affinity.PodAntiAffinity == nil && annotationsAffinity.PodAntiAffinity != nil {
				affinity.PodAntiAffinity = annotationsAffinity.PodAntiAffinity
			}
		}
	}
	return affinity
}
