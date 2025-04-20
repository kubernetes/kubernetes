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
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
)

// RecommendedDefaultDisruptionControllerConfiguration sets the defaults for DisruptionController.
func RecommendedDefaultDisruptionControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.DisruptionControllerConfiguration) {
	if obj.ConcurrentDisruptionSyncs == 0 {
		obj.ConcurrentDisruptionSyncs = 1
	}
	if obj.ConcurrentDisruptionStalePodSyncs == 0 {
		obj.ConcurrentDisruptionStalePodSyncs = 1
	}
}
