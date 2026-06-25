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

package configuration

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestExcludedResourcesNamedByRule(t *testing.T) {
	tests := []struct {
		name        string
		apiGroups   []string
		apiVersions []string
		resources   []string
		want        []schema.GroupResource
	}{
		{
			name:      "explicit excluded resource",
			apiGroups: []string{"authentication.k8s.io"}, apiVersions: []string{"v1"}, resources: []string{"tokenreviews"},
			want: []schema.GroupResource{{Group: "authentication.k8s.io", Resource: "tokenreviews"}},
		},
		{
			name:      "non-excluded resource",
			apiGroups: []string{""}, apiVersions: []string{"v1"}, resources: []string{"pods"},
			want: nil,
		},
		{
			name:      "wildcard group is not flagged",
			apiGroups: []string{"*"}, apiVersions: []string{"v1"}, resources: []string{"tokenreviews"},
			want: nil,
		},
		{
			name:      "wildcard version is not flagged",
			apiGroups: []string{"authentication.k8s.io"}, apiVersions: []string{"*"}, resources: []string{"tokenreviews"},
			want: nil,
		},
		{
			name:      "wildcard resource is not flagged",
			apiGroups: []string{"authentication.k8s.io"}, apiVersions: []string{"v1"}, resources: []string{"*"},
			want: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := excludedResourcesNamedByRule(tc.apiGroups, tc.apiVersions, tc.resources); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("excludedResourcesNamedByRule() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestLogExcludedResourcesForWebhook is a smoke test ensuring the logging helpers run without
// panicking when the feature is enabled.
func TestLogExcludedResourcesForWebhook(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExcludeAdmissionWebhookVirtualResources, true)

	logExcludedResourcesForValidatingWebhook("test-config", []v1.ValidatingWebhook{{
		Name:  "test-webhook",
		Rules: []v1.RuleWithOperations{{Rule: v1.Rule{APIGroups: []string{"authentication.k8s.io"}, APIVersions: []string{"v1"}, Resources: []string{"tokenreviews"}}}},
	}})
	logExcludedResourcesForMutatingWebhook("test-config", []v1.MutatingWebhook{{
		Name:  "test-mutating-webhook",
		Rules: []v1.RuleWithOperations{{Rule: v1.Rule{APIGroups: []string{"authorization.k8s.io"}, APIVersions: []string{"v1"}, Resources: []string{"subjectaccessreviews"}}}},
	}})
}
