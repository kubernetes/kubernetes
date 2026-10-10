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
	"k8s.io/kubernetes/pkg/apis/authorization"
)

// Funcs returns the fuzzer functions for the authorization api group.
//
// The v1beta1 schema lacks Spec.AuthorizationOptions and Status.ConditionalDecision,
// so the conversion to v1beta1 is lossy: AuthorizationOptions is dropped, and a
// ConditionalDecision is folded into a fail-closed unconditional Allowed/Denied status.
// Neither survives a round trip, so the fuzzer leaves them nil to keep cross-version
// round-trip tests from spuriously failing. Content-level fuzzing of those fields
// is exercised by the dedicated declarative-validation tests.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *authorization.SubjectAccessReviewSpec, c randfill.Continue) {
			c.FillNoCustom(s)
			s.AuthorizationOptions = nil
		},
		func(s *authorization.SelfSubjectAccessReviewSpec, c randfill.Continue) {
			c.FillNoCustom(s)
			s.AuthorizationOptions = nil
		},
		func(s *authorization.SubjectAccessReviewStatus, c randfill.Continue) {
			c.FillNoCustom(s)
			s.ConditionalDecision = nil
		},
	}
}
