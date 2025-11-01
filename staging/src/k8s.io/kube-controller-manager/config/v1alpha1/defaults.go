/*
Copyright 2025 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	scheme.AddTypeDefaultingFunc(&KubeControllerManagerConfiguration{}, func(obj interface{}) {
		SetObjectDefaultsKubeControllerManagerConfiguration(obj.(*KubeControllerManagerConfiguration))
	})
	return nil
}

func SetObjectDefaultsKubeControllerManagerConfiguration(in *KubeControllerManagerConfiguration) {
	SetDefaults_ResourceClaimControllerConfiguration(&in.ResourceClaimController)
}

// SetDefaults_ResourceClaimControllerConfiguration sets defaults for ResourceClaimController configuration.
func SetDefaults_ResourceClaimControllerConfiguration(obj *ResourceClaimControllerConfiguration) {
	if obj.ConcurrentResourceClaimSyncs == 0 {
		obj.ConcurrentResourceClaimSyncs = 50
	}
}
