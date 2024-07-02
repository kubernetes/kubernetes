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

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	nodeconfigv1alpha1 "k8s.io/cloud-provider/controllers/node/config/v1alpha1"
	serviceconfigv1alpha1 "k8s.io/cloud-provider/controllers/service/config/v1alpha1"
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
	// Use the default RecommendedDefaultNodeControllerConfiguration options
	nodeconfigv1alpha1.RecommendedDefaultNodeControllerConfiguration(&obj.NodeController)
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

// SetDefaults_ValidatingWebhook sets defaults for webhook validating. This function
// is duplicated from "k8s.io/kubernetes/pkg/apis/admissionregistration/v1/defaults.go"
// in order for in-tree cloud providers to not depend on internal packages.
func SetDefaults_ValidatingWebhook(obj *admissionregistrationv1.ValidatingWebhook) {
	if obj.FailurePolicy == nil {
		policy := admissionregistrationv1.Fail
		obj.FailurePolicy = &policy
	}
	if obj.MatchPolicy == nil {
		policy := admissionregistrationv1.Equivalent
		obj.MatchPolicy = &policy
	}
	if obj.NamespaceSelector == nil {
		selector := metav1.LabelSelector{}
		obj.NamespaceSelector = &selector
	}
	if obj.ObjectSelector == nil {
		selector := metav1.LabelSelector{}
		obj.ObjectSelector = &selector
	}
	if obj.TimeoutSeconds == nil {
		obj.TimeoutSeconds = new(int32)
		*obj.TimeoutSeconds = 10
	}
}

// SetDefaults_MutatingWebhook sets defaults for webhook mutating This function
// is duplicated from "k8s.io/kubernetes/pkg/apis/admissionregistration/v1/defaults.go"
// in order for in-tree cloud providers to not depend on internal packages.
func SetDefaults_MutatingWebhook(obj *admissionregistrationv1.MutatingWebhook) {
	if obj.FailurePolicy == nil {
		policy := admissionregistrationv1.Fail
		obj.FailurePolicy = &policy
	}
	if obj.MatchPolicy == nil {
		policy := admissionregistrationv1.Equivalent
		obj.MatchPolicy = &policy
	}
	if obj.NamespaceSelector == nil {
		selector := metav1.LabelSelector{}
		obj.NamespaceSelector = &selector
	}
	if obj.ObjectSelector == nil {
		selector := metav1.LabelSelector{}
		obj.ObjectSelector = &selector
	}
	if obj.TimeoutSeconds == nil {
		obj.TimeoutSeconds = new(int32)
		*obj.TimeoutSeconds = 10
	}
	if obj.ReinvocationPolicy == nil {
		never := admissionregistrationv1.NeverReinvocationPolicy
		obj.ReinvocationPolicy = &never
	}
}

// SetDefaults_Rule sets defaults for webhook rule This function
// is duplicated from "k8s.io/kubernetes/pkg/apis/admissionregistration/v1/defaults.go"
// in order for in-tree cloud providers to not depend on internal packages.
func SetDefaults_Rule(obj *admissionregistrationv1.Rule) {
	if obj.Scope == nil {
		s := admissionregistrationv1.AllScopes
		obj.Scope = &s
	}
}

// SetDefaults_ServiceReference sets defaults for Webhook's ServiceReference This function
// is duplicated from "k8s.io/kubernetes/pkg/apis/admissionregistration/v1/defaults.go"
// in order for in-tree cloud providers to not depend on internal packages.
func SetDefaults_ServiceReference(obj *admissionregistrationv1.ServiceReference) {
	if obj.Port == nil {
		obj.Port = utilpointer.Int32(443)
	}
}
