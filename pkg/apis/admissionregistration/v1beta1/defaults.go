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

package v1beta1

import (
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilpointer "k8s.io/utils/pointer"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_ValidatingWebhook sets defaults for webhook validating
func SetDefaults_ValidatingWebhook(obj *admissionregistrationv1beta1.ValidatingWebhook) {
	if obj.FailurePolicy == nil {
		policy := admissionregistrationv1beta1.Ignore
		obj.FailurePolicy = &policy
	}
	if obj.MatchPolicy == nil {
		policy := admissionregistrationv1beta1.Exact
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
	if obj.SideEffects == nil {
		// TODO: revisit/remove this default and possibly make the field required when promoting to v1
		unknown := admissionregistrationv1beta1.SideEffectClassUnknown
		obj.SideEffects = &unknown
	}
	if obj.TimeoutSeconds == nil {
		obj.TimeoutSeconds = new(int32)
		*obj.TimeoutSeconds = 30
	}

	if len(obj.AdmissionReviewVersions) == 0 {
		obj.AdmissionReviewVersions = []string{admissionregistrationv1beta1.SchemeGroupVersion.Version}
	}
}

// SetDefaults_MutatingWebhook sets defaults for webhook mutating
func SetDefaults_MutatingWebhook(obj *admissionregistrationv1beta1.MutatingWebhook) {
	if obj.FailurePolicy == nil {
		policy := admissionregistrationv1beta1.Ignore
		obj.FailurePolicy = &policy
	}
	if obj.MatchPolicy == nil {
		policy := admissionregistrationv1beta1.Exact
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
	if obj.SideEffects == nil {
		// TODO: revisit/remove this default and possibly make the field required when promoting to v1
		unknown := admissionregistrationv1beta1.SideEffectClassUnknown
		obj.SideEffects = &unknown
	}
	if obj.TimeoutSeconds == nil {
		obj.TimeoutSeconds = new(int32)
		*obj.TimeoutSeconds = 30
	}
	if obj.ReinvocationPolicy == nil {
		never := admissionregistrationv1beta1.NeverReinvocationPolicy
		obj.ReinvocationPolicy = &never
	}

	if len(obj.AdmissionReviewVersions) == 0 {
		obj.AdmissionReviewVersions = []string{admissionregistrationv1beta1.SchemeGroupVersion.Version}
	}
}

// SetDefaults_Rule sets defaults for webhook rule
func SetDefaults_Rule(obj *admissionregistrationv1beta1.Rule) {
	if obj.Scope == nil {
		s := admissionregistrationv1beta1.AllScopes
		obj.Scope = &s
	}
}

// SetDefaults_ServiceReference sets defaults for Webhook's ServiceReference
func SetDefaults_ServiceReference(obj *admissionregistrationv1beta1.ServiceReference) {
	if obj.Port == nil {
		obj.Port = utilpointer.Int32Ptr(443)
	}
}
