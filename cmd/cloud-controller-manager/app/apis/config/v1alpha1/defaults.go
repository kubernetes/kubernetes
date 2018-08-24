/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	ctrlmgrconfigv1alpha1 "k8s.io/controller-manager/pkg/apis/config/v1alpha1"
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_CloudControllerManagerConfiguration(obj *CloudControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.NodeStatusUpdateFrequency == zero {
		obj.NodeStatusUpdateFrequency = metav1.Duration{Duration: 5 * time.Minute}
	}

	if obj.Generic.ClientConnection.QPS == 0.0 {
		obj.Generic.ClientConnection.QPS = 20.0
	}

	if obj.Generic.ClientConnection.Burst == 0 {
		obj.Generic.ClientConnection.Burst = 30
	}

	// Use the default GenericControllerManagerConfiguration
	ctrlmgrconfigv1alpha1.RecommendedDefaultGenericControllerManagerConfiguration(&obj.Generic)
}
