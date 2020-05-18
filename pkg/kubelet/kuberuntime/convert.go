/*
Copyright 2020 The Kubernetes Authors.

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

package kuberuntime

import (
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// This file contains help function to kuberuntime types to CRI runtime API types, or vice versa.

func toKubeContainerImageSpec(image *runtimeapi.Image) kubecontainer.ImageSpec {
	var annotations []kubecontainer.Annotation

	if image.Spec != nil && image.Spec.Annotations != nil {
		for k, v := range image.Spec.Annotations {
			annotations = append(annotations, kubecontainer.Annotation{
				Name:  k,
				Value: v,
			})
		}
	}

	return kubecontainer.ImageSpec{
		Image:       image.Id,
		Annotations: annotations,
	}
}

func toRuntimeAPIImageSpec(imageSpec kubecontainer.ImageSpec) *runtimeapi.ImageSpec {
	var annotations = make(map[string]string)
	if imageSpec.Annotations != nil {
		for _, a := range imageSpec.Annotations {
			annotations[a.Name] = a.Value
		}
	}
	return &runtimeapi.ImageSpec{
		Image:       imageSpec.Image,
		Annotations: annotations,
	}
}
