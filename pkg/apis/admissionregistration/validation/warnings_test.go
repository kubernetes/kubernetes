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

package validation

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

// testExcludedResources is a representative excluded set; the production set is supplied by the
// caller (see pkg/kubeapiserver/admission/exclusion), so these helpers don't depend on it.
var testExcludedResources = []schema.GroupResource{
	{Group: "authentication.k8s.io", Resource: "tokenreviews"},
	{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"},
}

func rule(groups, versions, resources []string) admissionregistration.RuleWithOperations {
	return admissionregistration.RuleWithOperations{
		Rule: admissionregistration.Rule{
			APIGroups:   groups,
			APIVersions: versions,
			Resources:   resources,
		},
	}
}

func TestWarningsForWebhookRules(t *testing.T) {
	tests := []struct {
		name     string
		rules    []admissionregistration.RuleWithOperations
		wantWarn []string
	}{
		{
			name:     "no warnings for normal resources",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{""}, []string{"v1"}, []string{"pods", "configmaps"})},
			wantWarn: nil,
		},
		{
			name:  "warning for explicit excluded resource",
			rules: []admissionregistration.RuleWithOperations{rule([]string{"authentication.k8s.io"}, []string{"v1"}, []string{"tokenreviews"})},
			wantWarn: []string{
				`webhooks[0].rules[0]: tokenreviews.authentication.k8s.io is excluded from admission webhooks; this rule will have no effect`,
			},
		},
		{
			name:  "warning only for the excluded resource in a mixed rule",
			rules: []admissionregistration.RuleWithOperations{rule([]string{"authorization.k8s.io"}, []string{"v1"}, []string{"subjectaccessreviews", "somethingelse"})},
			wantWarn: []string{
				`webhooks[0].rules[0]: subjectaccessreviews.authorization.k8s.io is excluded from admission webhooks; this rule will have no effect`,
			},
		},
		{
			name:     "no warning for all wildcards",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{"*"}, []string{"*"}, []string{"*"})},
			wantWarn: nil,
		},
		{
			name:     "no warning for wildcard apiGroups",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{"*"}, []string{"v1"}, []string{"tokenreviews"})},
			wantWarn: nil,
		},
		{
			name:     "no warning for wildcard apiVersions",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{"authentication.k8s.io"}, []string{"*"}, []string{"tokenreviews"})},
			wantWarn: nil,
		},
		{
			name:     "no warning for wildcard resources",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{"authentication.k8s.io"}, []string{"v1"}, []string{"*"})},
			wantWarn: nil,
		},
		{
			name:     "no warning for excluded resource in wrong group",
			rules:    []admissionregistration.RuleWithOperations{rule([]string{"example.com"}, []string{"v1"}, []string{"tokenreviews"})},
			wantWarn: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			validating := []admissionregistration.ValidatingWebhook{{Name: "w", Rules: tc.rules}}
			if got := WarningsForValidatingWebhookRules(validating, testExcludedResources); !reflect.DeepEqual(got, tc.wantWarn) {
				t.Errorf("WarningsForValidatingWebhookRules() = %v, want %v", got, tc.wantWarn)
			}
			mutating := []admissionregistration.MutatingWebhook{{Name: "w", Rules: tc.rules}}
			if got := WarningsForMutatingWebhookRules(mutating, testExcludedResources); !reflect.DeepEqual(got, tc.wantWarn) {
				t.Errorf("WarningsForMutatingWebhookRules() = %v, want %v", got, tc.wantWarn)
			}
		})
	}
}
