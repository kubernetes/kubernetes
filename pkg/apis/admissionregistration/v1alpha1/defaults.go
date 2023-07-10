/*
Copyright 2022 The Kubernetes Authors.

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
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_ValidatingAdmissionPolicySpec sets defaults for ValidatingAdmissionPolicySpec
func SetDefaults_ValidatingAdmissionPolicySpec(obj *admissionregistrationv1alpha1.ValidatingAdmissionPolicySpec) {
	if obj.FailurePolicy == nil {
		policy := admissionregistrationv1alpha1.Fail
		obj.FailurePolicy = &policy
	}
}

// SetDefaults_MatchResources sets defaults for MatchResources
func SetDefaults_MatchResources(obj *admissionregistrationv1alpha1.MatchResources) {
	if obj.MatchPolicy == nil {
		policy := admissionregistrationv1alpha1.Equivalent
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
}

// SetDefaults_ParamRef sets defaults for ParamRef
func SetDefaults_ParamRef(obj *admissionregistrationv1alpha1.ParamRef) {
	if obj.ParameterNotFoundAction == nil {
		v := admissionregistrationv1alpha1.DenyAction
		obj.ParameterNotFoundAction = &v
	}
}
