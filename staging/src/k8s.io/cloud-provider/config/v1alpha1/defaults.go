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
	"k8s.io/apimachinery/pkg/runtime"
	serviceconfigv1alpha1 "k8s.io/cloud-provider/service/config/v1alpha1"
	cmconfigv1alpha1 "k8s.io/controller-manager/config/v1alpha1"
	utilpointer "k8s.io/utils/pointer"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_CloudControllerManagerConfiguration(obj *CloudControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.NodeStatusUpdateFrequency == zero {
		obj.NodeStatusUpdateFrequency = metav1.Duration{Duration: 5 * time.Minute}
	}

	// These defaults override the recommended defaults from the apimachineryconfigv1alpha1 package that are applied automatically
	// These client-connection defaults are specific to the cloud-controller-manager
	if obj.Generic.ClientConnection.QPS == 0 {
		obj.Generic.ClientConnection.QPS = 20
	}
	if obj.Generic.ClientConnection.Burst == 0 {
		obj.Generic.ClientConnection.Burst = 30
	}

	// Use the default RecommendedDefaultGenericControllerManagerConfiguration options
	cmconfigv1alpha1.RecommendedDefaultGenericControllerManagerConfiguration(&obj.Generic)
	// Use the default RecommendedDefaultServiceControllerConfiguration options
	serviceconfigv1alpha1.RecommendedDefaultServiceControllerConfiguration(&obj.ServiceController)
}

func SetDefaults_KubeCloudSharedConfiguration(obj *KubeCloudSharedConfiguration) {
	zero := metav1.Duration{}
	if obj.NodeMonitorPeriod == zero {
		obj.NodeMonitorPeriod = metav1.Duration{Duration: 5 * time.Second}
	}
	if obj.ClusterName == "" {
		obj.ClusterName = "kubernetes"
	}
	if obj.ConfigureCloudRoutes == nil {
		obj.ConfigureCloudRoutes = utilpointer.BoolPtr(true)
	}
	if obj.RouteReconciliationPeriod == zero {
		obj.RouteReconciliationPeriod = metav1.Duration{Duration: 10 * time.Second}
	}
}
