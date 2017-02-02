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
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

func GetPodTemplateWithHash(ds *extensions.DaemonSet, dsPodTemplateSpecHash uint32) v1.PodTemplateSpec {
	obj, _ := api.Scheme.DeepCopy(ds.Spec.Template)
	dsPodTemplateSpecHashStr := strconv.FormatUint(uint64(dsPodTemplateSpecHash), 10)
	template := obj.(v1.PodTemplateSpec)
	template.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		ds.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDaemonSetUniqueLabelKey,
		dsPodTemplateSpecHashStr,
	)
	return template
}

func IsPodUpdated(dsPodTemplateSpecHash uint32, pod *v1.Pod) bool {
	curPodTemplateSpecHash, hashExists := pod.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
	dsPodTemplateSpecHashStr := strconv.FormatUint(uint64(dsPodTemplateSpecHash), 10)
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
