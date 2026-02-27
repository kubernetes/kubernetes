/*
Copyright The Kubernetes Authors.

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

package v1alpha2

import (
	"k8s.io/api/scheduling/v1alpha2"
	runtime "k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_PodGroupSpec sets defaults for PodGroupSpec.
func SetDefaults_PodGroupSpec(obj *v1alpha2.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) && obj.DisruptionMode == nil {
		obj.DisruptionMode = new(v1alpha2.DisruptionModePod)
	}
}

// SetDefaults_PodGroupTemplate sets defaults for PodGroupTemplate.
func SetDefaults_PodGroupTemplate(obj *v1alpha2.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) && obj.DisruptionMode == nil {
		obj.DisruptionMode = new(v1alpha2.DisruptionModePod)
	}
}
