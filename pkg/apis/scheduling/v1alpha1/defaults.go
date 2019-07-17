/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_PriorityClass sets additional defaults compared to its counterpart
// in extensions.
func SetDefaults_PriorityClass(obj *v1alpha1.PriorityClass) {
	if utilfeature.DefaultFeatureGate.Enabled(features.NonPreemptingPriority) && obj.PreemptionPolicy == nil {
		preemptLowerPriority := apiv1.PreemptLowerPriority
		obj.PreemptionPolicy = &preemptLowerPriority
	}
}
