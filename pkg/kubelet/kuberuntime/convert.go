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
	"sort"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// This file contains help function to kuberuntime types to CRI runtime API types, or vice versa.

func toKubeContainerImageSpec(image *runtimeapi.Image) kubecontainer.ImageSpec {
	var annotations []kubecontainer.Annotation

	if image.Spec != nil && len(image.Spec.Annotations) > 0 {
		annotationKeys := make([]string, 0, len(image.Spec.Annotations))
		for k := range image.Spec.Annotations {
			annotationKeys = append(annotationKeys, k)
		}
		sort.Strings(annotationKeys)
		for _, k := range annotationKeys {
			annotations = append(annotations, kubecontainer.Annotation{
				Name:  k,
				Value: image.Spec.Annotations[k],
			})
		}
	}

	spec := kubecontainer.ImageSpec{
		Image:       image.Id,
		Annotations: annotations,
	}
	// if RuntimeClassInImageCriAPI feature gate is enabled, set runtimeHandler CRI field
	if utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClassInImageCriAPI) {
		runtimeHandler := ""
		if image.Spec != nil {
			runtimeHandler = image.Spec.RuntimeHandler
		}
		spec.RuntimeHandler = runtimeHandler
	}

	return spec
}

func toRuntimeAPIImageSpec(imageSpec kubecontainer.ImageSpec) *runtimeapi.ImageSpec {
	var annotations = make(map[string]string)
	if imageSpec.Annotations != nil {
		for _, a := range imageSpec.Annotations {
			annotations[a.Name] = a.Value
		}
	}

	spec := runtimeapi.ImageSpec{
		Image:       imageSpec.Image,
		Annotations: annotations,
	}
	// if RuntimeClassInImageCriAPI feature gate is enabled, set runtimeHandler CRI field
	if utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClassInImageCriAPI) {
		spec.RuntimeHandler = imageSpec.RuntimeHandler
	}

	return &spec
}
