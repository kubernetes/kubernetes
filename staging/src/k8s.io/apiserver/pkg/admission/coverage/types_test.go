/*
Copyright The Kubernetes Authors.

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

package coverage

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

func TestViolationStrings(t *testing.T) {
	pods := schema.GroupResource{Resource: "pods"}
	deployments := schema.GroupResource{Group: "apps", Resource: "deployments"}
	testcases := []struct {
		name       string
		v          Violation
		wantTarget string
		wantSource string
		wantRule   string
	}{
		{
			name:       "subresource target",
			v:          Violation{Resource: pods, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}, Source: Shape{Operation: admission.Create}},
			wantTarget: "UPDATE pods/dynamic",
			wantSource: "CREATE pods",
			wantRule:   `a rule for UPDATE on "pods/dynamic" (or "pods/*")`,
		},
		{
			name:       "main resource target",
			v:          Violation{Resource: pods, Target: Shape{Operation: admission.Update}, Source: Shape{Operation: admission.Update, Subresource: "resize"}},
			wantTarget: "UPDATE pods",
			wantSource: "UPDATE pods/resize",
			wantRule:   `a rule for UPDATE on "pods" (or "pods/*")`,
		},
		{
			name:       "non-core group",
			v:          Violation{Resource: deployments, Target: Shape{Operation: admission.Update, Subresource: "scale"}, Source: Shape{Operation: admission.Update}},
			wantTarget: "UPDATE deployments.apps/scale",
			wantSource: "UPDATE deployments.apps",
			wantRule:   `a rule for UPDATE on "deployments/scale" (or "deployments/*")`,
		},
		{
			// Built by callers when selector evaluation fails, so the source is unknown.
			name:       "empty source",
			v:          Violation{Resource: pods, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}},
			wantTarget: "UPDATE pods/dynamic",
			wantSource: "",
			wantRule:   `a rule for UPDATE on "pods/dynamic" (or "pods/*")`,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.v.TargetString(); got != tc.wantTarget {
				t.Errorf("TargetString() = %q, want %q", got, tc.wantTarget)
			}
			if got := tc.v.SourceString(); got != tc.wantSource {
				t.Errorf("SourceString() = %q, want %q", got, tc.wantSource)
			}
			if got := tc.v.RuleHint(); got != tc.wantRule {
				t.Errorf("RuleHint() = %q, want %q", got, tc.wantRule)
			}
		})
	}
}
