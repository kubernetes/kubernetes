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
	"k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

// SetDefaults_FlowSchema sets default values for flow schema
func SetDefaults_FlowSchema(obj *v1alpha1.FlowSchema) {
	if obj.Spec.MatchingPrecedence == 0 {
		obj.Spec.MatchingPrecedence = flowcontrol.FlowSchemaDefaultMatchingPrecedence
	}
}

// SetDefaults_FlowSchema sets default values for flow schema
func SetDefaults_PriorityLevelConfiguration(obj *v1alpha1.PriorityLevelConfiguration) {
	if !obj.Spec.Exempt {
		if obj.Spec.HandSize == 0 {
			obj.Spec.HandSize = flowcontrol.PriorityLevelConfigurationDefaultHandSize
		}
	}
}

// SetDefaults_Subject defaults fields for subject
func SetDefaults_Subject(obj *v1alpha1.Subject) {
	if len(obj.APIGroup) == 0 {
		switch obj.Kind {
		case v1alpha1.ServiceAccountKind:
			// do nothing
		case v1alpha1.UserKind:
			obj.APIGroup = GroupName
		case v1alpha1.GroupKind:
			obj.APIGroup = GroupName
		}
	}
}
