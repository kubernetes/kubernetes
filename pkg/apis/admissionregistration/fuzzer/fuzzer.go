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

package fuzzer

import (
	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"

	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

// Funcs returns the fuzzer functions for the admissionregistration api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *admissionregistration.Rule, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.Scope == nil {
				s := admissionregistration.AllScopes
				obj.Scope = &s
			}
		},
		func(obj *admissionregistration.ValidatingWebhook, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.FailurePolicy == nil {
				p := admissionregistration.FailurePolicyType("Fail")
				obj.FailurePolicy = &p
			}
			if obj.MatchPolicy == nil {
				m := admissionregistration.MatchPolicyType("Exact")
				obj.MatchPolicy = &m
			}
			if obj.SideEffects == nil {
				s := admissionregistration.SideEffectClassUnknown
				obj.SideEffects = &s
			}
			if obj.TimeoutSeconds == nil {
				i := int32(30)
				obj.TimeoutSeconds = &i
			}
			obj.AdmissionReviewVersions = []string{"v1beta1"}
		},
		func(obj *admissionregistration.MutatingWebhook, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.FailurePolicy == nil {
				p := admissionregistration.FailurePolicyType("Fail")
				obj.FailurePolicy = &p
			}
			if obj.MatchPolicy == nil {
				m := admissionregistration.MatchPolicyType("Exact")
				obj.MatchPolicy = &m
			}
			if obj.SideEffects == nil {
				s := admissionregistration.SideEffectClassUnknown
				obj.SideEffects = &s
			}
			if obj.ReinvocationPolicy == nil {
				r := admissionregistration.NeverReinvocationPolicy
				obj.ReinvocationPolicy = &r
			}
			if obj.TimeoutSeconds == nil {
				i := int32(30)
				obj.TimeoutSeconds = &i
			}
			obj.AdmissionReviewVersions = []string{"v1beta1"}
		},
		func(obj *admissionregistration.ValidatingAdmissionPolicySpec, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.FailurePolicy == nil {
				p := admissionregistration.FailurePolicyType("Fail")
				obj.FailurePolicy = &p
			}
		},
		func(obj *admissionregistration.ValidatingAdmissionPolicyBindingSpec, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.ValidationActions == nil {
				obj.ValidationActions = []admissionregistration.ValidationAction{admissionregistration.Deny}
			}
		},
		func(obj *admissionregistration.MatchResources, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.MatchPolicy == nil {
				m := admissionregistration.MatchPolicyType("Exact")
				obj.MatchPolicy = &m
			}
		},
		func(obj *admissionregistration.ParamRef, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again

			// Populate required field
			if obj.ParameterNotFoundAction == nil {
				v := admissionregistration.DenyAction
				obj.ParameterNotFoundAction = &v
			}
		},
		func(obj *admissionregistration.MutatingAdmissionPolicySpec, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			if obj.FailurePolicy == nil {
				p := admissionregistration.FailurePolicyType("Fail")
				obj.FailurePolicy = &p
			}
			obj.ReinvocationPolicy = admissionregistration.NeverReinvocationPolicy
		},
		func(obj *admissionregistration.Mutation, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again
			patchTypes := []admissionregistration.PatchType{admissionregistration.PatchTypeJSONPatch, admissionregistration.PatchTypeApplyConfiguration}
			obj.PatchType = patchTypes[c.Rand.Intn(len(patchTypes))]
			if obj.PatchType == admissionregistration.PatchTypeJSONPatch {
				obj.JSONPatch = &admissionregistration.JSONPatch{}
				c.Fill(&obj.JSONPatch)
				obj.ApplyConfiguration = nil
			}
			if obj.PatchType == admissionregistration.PatchTypeApplyConfiguration {
				obj.ApplyConfiguration = &admissionregistration.ApplyConfiguration{}
				c.Fill(obj.ApplyConfiguration)
				obj.JSONPatch = nil
			}
		},
	}
}
