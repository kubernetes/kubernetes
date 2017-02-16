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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

// GetPodTemplateWithHash returns copy of provided template with additional
// label which contains hash of provided template
func GetPodTemplateWithGeneration(template v1.PodTemplateSpec, generation int64) v1.PodTemplateSpec {
	obj, _ := api.Scheme.DeepCopy(template)
	newTemplate := obj.(v1.PodTemplateSpec)
	templateGenerationStr := fmt.Sprint(generation)
	newTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		template.ObjectMeta.Labels,
		extensions.DaemonSetTemplateGenerationKey,
		templateGenerationStr,
	)
	return newTemplate
}

// IsPodUpdate checks if pod contains label with provided hash
func IsPodUpdated(dsTemplateGeneration int64, pod *v1.Pod) bool {
	podTemplateGeneration, generationExists := pod.ObjectMeta.Labels[extensions.DaemonSetTemplateGenerationKey]
	dsTemplateGenerationStr := fmt.Sprint(dsTemplateGeneration)
	return generationExists && podTemplateGeneration == dsTemplateGenerationStr
}

// SplitByAvailablePods splits provided daemon set pods by availabilty
func SplitByAvailablePods(minReadySeconds int32, pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod) {
	unavailablePods := []*v1.Pod{}
	availablePods := []*v1.Pod{}
	for _, pod := range pods {
		if v1.IsPodAvailable(pod, minReadySeconds, metav1.Now()) {
			availablePods = append(availablePods, pod)
		} else {
			unavailablePods = append(unavailablePods, pod)
		}
	}
	return availablePods, unavailablePods
}
