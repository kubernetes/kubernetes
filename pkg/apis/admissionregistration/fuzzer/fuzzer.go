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
	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

// Funcs returns the fuzzer functions for the admissionregistration api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *admissionregistration.Rule, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			if obj.Scope == nil {
				s := admissionregistration.AllScopes
				obj.Scope = &s
			}
		},
		func(obj *admissionregistration.ValidatingWebhook, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			p := admissionregistration.FailurePolicyType("Fail")
			obj.FailurePolicy = &p
			m := admissionregistration.MatchPolicyType("Exact")
			obj.MatchPolicy = &m
			s := admissionregistration.SideEffectClassUnknown
			obj.SideEffects = &s
			if obj.TimeoutSeconds == nil {
				i := int32(30)
				obj.TimeoutSeconds = &i
			}
			obj.AdmissionReviewVersions = []string{"v1beta1"}
		},
		func(obj *admissionregistration.MutatingWebhook, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			p := admissionregistration.FailurePolicyType("Fail")
			obj.FailurePolicy = &p
			m := admissionregistration.MatchPolicyType("Exact")
			obj.MatchPolicy = &m
			s := admissionregistration.SideEffectClassUnknown
			obj.SideEffects = &s
			n := admissionregistration.NeverReinvocationPolicy
			obj.ReinvocationPolicy = &n
			if obj.TimeoutSeconds == nil {
				i := int32(30)
				obj.TimeoutSeconds = &i
			}
			obj.AdmissionReviewVersions = []string{"v1beta1"}
		},
	}
}
