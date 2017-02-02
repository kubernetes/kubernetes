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
	podutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

// GetPodTemplateWithHash returns copy of provided template with additional
// label which contains hash of provided template
func GetPodTemplateWithHash(template v1.PodTemplateSpec) v1.PodTemplateSpec {
	dsPodTemplateSpecHash := podutil.GetPodTemplateSpecHashFnv(template)
	obj, _ := api.Scheme.DeepCopy(template)
	dsPodTemplateSpecHashStr := fmt.Sprint(dsPodTemplateSpecHash)
	newTemplate := obj.(v1.PodTemplateSpec)
	newTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		template.ObjectMeta.Labels,
		extensions.DefaultDaemonSetUniqueLabelKey,
		dsPodTemplateSpecHashStr,
	)
	return newTemplate
}

// IsPodUpdate checks if pod contains label with provided hash
func IsPodUpdated(dsPodTemplateSpecHash uint32, pod *v1.Pod) bool {
	curPodTemplateSpecHash, hashExists := pod.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
	dsPodTemplateSpecHashStr := fmt.Sprint(dsPodTemplateSpecHash)
	if hashExists {
		if curPodTemplateSpecHash == dsPodTemplateSpecHashStr {
			return true
		}
	} else {
		// XXX: Hash does not exist. It's old pod. For now returning true.
		return true
	}
	return false
}

// SplitByAvailablePods splits provided daemon set pods by availabilty
func SplitByAvailablePods(ds *extensions.DaemonSet, pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod) {
	unavailablePods := []*v1.Pod{}
	availablePods := []*v1.Pod{}
	for _, pod := range pods {
		if v1.IsPodAvailable(pod, ds.Spec.MinReadySeconds, metav1.Now()) {
			availablePods = append(availablePods, pod)
		} else {
			unavailablePods = append(unavailablePods, pod)
		}
	}
	return availablePods, unavailablePods
}
