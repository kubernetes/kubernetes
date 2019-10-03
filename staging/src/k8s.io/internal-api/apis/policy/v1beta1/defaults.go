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

package v1beta1

import (
	policyv1beta1 "k8s.io/api/policy/v1beta1"
)

func SetDefaults_PodSecurityPolicySpec(obj *policyv1beta1.PodSecurityPolicySpec) {
	// This field was added after PodSecurityPolicy was released.
	// Policies that do not include this field must remain as permissive as they were prior to the introduction of this field.
	if obj.AllowPrivilegeEscalation == nil {
		t := true
		obj.AllowPrivilegeEscalation = &t
	}
}
