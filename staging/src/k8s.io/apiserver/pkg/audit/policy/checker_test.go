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

package policy

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

var (
	tim = &user.DefaultInfo{
		Name:   "tim@k8s.io",
		Groups: []string{"humans", "developers"},
	}
	attrs = map[string]authorizer.Attributes{
		"namespaced": &authorizer.AttributesRecord{
			User:            tim,
			Verb:            "get",
			Namespace:       "default",
			APIGroup:        "", // Core
			APIVersion:      "v1",
			Resource:        "pods",
			Name:            "busybox",
			ResourceRequest: true,
			Path:            "/api/v1/namespaces/default/pods/busybox",
		},
		"cluster": &authorizer.AttributesRecord{
			User:            tim,
			Verb:            "get",
			APIGroup:        "rbac.authorization.k8s.io", // Core
			APIVersion:      "v1beta1",
			Resource:        "clusterroles",
			Name:            "edit",
			ResourceRequest: true,
			Path:            "/apis/rbac.authorization.k8s.io/v1beta1/clusterroles/edit",
		},
		"nonResource": &authorizer.AttributesRecord{
			User:            tim,
			Verb:            "get",
			ResourceRequest: false,
			Path:            "/logs/kubelet.log",
		},
		"subresource": &authorizer.AttributesRecord{
			User:            tim,
			Verb:            "get",
			Namespace:       "default",
			APIGroup:        "", // Core
			APIVersion:      "v1",
			Resource:        "pods",
			Subresource:     "log",
			Name:            "busybox",
			ResourceRequest: true,
			Path:            "/api/v1/namespaces/default/pods/busybox",
		},
		"Unauthorized": &authorizer.AttributesRecord{
			Verb:            "get",
			Namespace:       "default",
			APIGroup:        "", // Core
			APIVersion:      "v1",
			Resource:        "pods",
			Name:            "busybox",
			ResourceRequest: true,
			Path:            "/api/v1/namespaces/default/pods/busybox",
		},
	}

	rules = map[string]audit.PolicyRule{
		"default": {
			Level: audit.LevelMetadata,
		},
		"create": {
			Level: audit.LevelRequest,
			Verbs: []string{"create"},
		},
		"tims": {
			Level: audit.LevelMetadata,
			Users: []string{"tim@k8s.io"},
		},
		"humans": {
			Level:      audit.LevelMetadata,
			UserGroups: []string{"humans"},
		},
		"serviceAccounts": {
			Level:      audit.LevelRequest,
			UserGroups: []string{"system:serviceaccounts"},
		},
		"getPods": {
			Level:     audit.LevelRequestResponse,
			Verbs:     []string{"get"},
			Resources: []audit.GroupResources{{Resources: []string{"pods"}}},
		},
		"getPodLogs": {
			Level:     audit.LevelRequest,
			Verbs:     []string{"get"},
			Resources: []audit.GroupResources{{Resources: []string{"pods/log"}}},
		},
		"getPodWildcardMatching": {
			Level:     audit.LevelRequest,
			Verbs:     []string{"get"},
			Resources: []audit.GroupResources{{Resources: []string{"*"}}},
		},
		"getPodResourceWildcardMatching": {
			Level:     audit.LevelRequest,
			Verbs:     []string{"get"},
			Resources: []audit.GroupResources{{Resources: []string{"*/log"}}},
		},
		"getPodSubResourceWildcardMatching": {
			Level:     audit.LevelRequest,
			Verbs:     []string{"get"},
			Resources: []audit.GroupResources{{Resources: []string{"pods/*"}}},
		},
		"getClusterRoles": {
			Level: audit.LevelRequestResponse,
			Verbs: []string{"get"},
			Resources: []audit.GroupResources{{
				Group:     "rbac.authorization.k8s.io",
				Resources: []string{"clusterroles"},
			}},
			Namespaces: []string{""},
		},
		"getLogs": {
			Level: audit.LevelRequestResponse,
			Verbs: []string{"get"},
			NonResourceURLs: []string{
				"/logs*",
			},
		},
		"getMetrics": {
			Level: audit.LevelRequest,
			Verbs: []string{"get"},
			NonResourceURLs: []string{
				"/metrics",
			},
		},
		"clusterRoleEdit": {
			Level: audit.LevelRequest,
			Resources: []audit.GroupResources{{
				Group:         "rbac.authorization.k8s.io",
				Resources:     []string{"clusterroles"},
				ResourceNames: []string{"edit"},
			}},
		},
		"omit RequestReceived": {
			Level: audit.LevelRequest,
			OmitStages: []audit.Stage{
				audit.StageRequestReceived,
			},
		},
		"only audit panic": {
			Level: audit.LevelRequest,
			OmitStages: []audit.Stage{
				audit.StageRequestReceived,
				audit.StageResponseStarted,
				audit.StageResponseComplete,
			},
		},
	}
)

func test(t *testing.T, req string, expLevel audit.Level, policyStages, expOmitStages []audit.Stage, ruleNames ...string) {
	policy := audit.Policy{OmitStages: policyStages}
	for _, rule := range ruleNames {
		require.Contains(t, rules, rule)
		policy.Rules = append(policy.Rules, rules[rule])
	}
	require.Contains(t, attrs, req)
	auditConfig := NewPolicyRuleEvaluator(&policy).EvaluatePolicyRule(attrs[req])
	assert.Equal(t, expLevel, auditConfig.Level, "request:%s rules:%s", req, strings.Join(ruleNames, ","))
	assert.True(t, stageEqual(expOmitStages, auditConfig.OmitStages), "request:%s rules:%s, expected stages: %v, actual stages: %v",
		req, strings.Join(ruleNames, ","), expOmitStages, auditConfig.OmitStages)
}

func testAuditLevel(t *testing.T, stages []audit.Stage) {
	test(t, "namespaced", audit.LevelMetadata, stages, stages, "default")
	test(t, "namespaced", audit.LevelNone, stages, stages, "create")
	test(t, "namespaced", audit.LevelMetadata, stages, stages, "tims")
	test(t, "namespaced", audit.LevelMetadata, stages, stages, "humans")
	test(t, "namespaced", audit.LevelNone, stages, stages, "serviceAccounts")
	test(t, "namespaced", audit.LevelRequestResponse, stages, stages, "getPods")
	test(t, "namespaced", audit.LevelNone, stages, stages, "getClusterRoles")
	test(t, "namespaced", audit.LevelNone, stages, stages, "getLogs")
	test(t, "namespaced", audit.LevelNone, stages, stages, "getMetrics")
	test(t, "namespaced", audit.LevelMetadata, stages, stages, "getMetrics", "serviceAccounts", "default")
	test(t, "namespaced", audit.LevelRequestResponse, stages, stages, "getMetrics", "getPods", "default")
	test(t, "namespaced", audit.LevelRequestResponse, stages, stages, "getPodLogs", "getPods")

	test(t, "cluster", audit.LevelMetadata, stages, stages, "default")
	test(t, "cluster", audit.LevelNone, stages, stages, "create")
	test(t, "cluster", audit.LevelMetadata, stages, stages, "tims")
	test(t, "cluster", audit.LevelMetadata, stages, stages, "humans")
	test(t, "cluster", audit.LevelNone, stages, stages, "serviceAccounts")
	test(t, "cluster", audit.LevelNone, stages, stages, "getPods")
	test(t, "cluster", audit.LevelRequestResponse, stages, stages, "getClusterRoles")
	test(t, "cluster", audit.LevelRequest, stages, stages, "clusterRoleEdit", "getClusterRoles")
	test(t, "cluster", audit.LevelNone, stages, stages, "getLogs")
	test(t, "cluster", audit.LevelNone, stages, stages, "getMetrics")
	test(t, "cluster", audit.LevelMetadata, stages, stages, "getMetrics", "serviceAccounts", "default")
	test(t, "cluster", audit.LevelRequestResponse, stages, stages, "getMetrics", "getClusterRoles", "default")
	test(t, "cluster", audit.LevelNone, stages, stages, "getPodLogs", "getPods")

	test(t, "nonResource", audit.LevelMetadata, stages, stages, "default")
	test(t, "nonResource", audit.LevelNone, stages, stages, "create")
	test(t, "nonResource", audit.LevelMetadata, stages, stages, "tims")
	test(t, "nonResource", audit.LevelMetadata, stages, stages, "humans")
	test(t, "nonResource", audit.LevelNone, stages, stages, "serviceAccounts")
	test(t, "nonResource", audit.LevelNone, stages, stages, "getPods")
	test(t, "nonResource", audit.LevelNone, stages, stages, "getClusterRoles")
	test(t, "nonResource", audit.LevelRequestResponse, stages, stages, "getLogs")
	test(t, "nonResource", audit.LevelNone, stages, stages, "getMetrics")
	test(t, "nonResource", audit.LevelMetadata, stages, stages, "getMetrics", "serviceAccounts", "default")
	test(t, "nonResource", audit.LevelRequestResponse, stages, stages, "getLogs", "getClusterRoles", "default")
	test(t, "nonResource", audit.LevelNone, stages, stages, "getPodLogs", "getPods")

	test(t, "subresource", audit.LevelRequest, stages, stages, "getPodLogs", "getPods")
	test(t, "subresource", audit.LevelRequest, stages, stages, "getPodWildcardMatching")
	test(t, "subresource", audit.LevelRequest, stages, stages, "getPodResourceWildcardMatching")
	test(t, "subresource", audit.LevelRequest, stages, stages, "getPodSubResourceWildcardMatching")

	test(t, "Unauthorized", audit.LevelNone, stages, stages, "tims")
	test(t, "Unauthorized", audit.LevelMetadata, stages, stages, "tims", "default")
	test(t, "Unauthorized", audit.LevelNone, stages, stages, "humans")
	test(t, "Unauthorized", audit.LevelMetadata, stages, stages, "humans", "default")
}

func TestChecker(t *testing.T) {
	testAuditLevel(t, nil)

	// test omitStages pre rule
	test(t, "namespaced", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test(t, "namespaced", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
	test(t, "cluster", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test(t, "cluster", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
	test(t, "nonResource", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test(t, "nonResource", audit.LevelRequest, nil, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
}

func TestCheckerPolicyOmitStages(t *testing.T) {
	policyStages := []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted}
	testAuditLevel(t, policyStages)

	// test omitStages policy wide
	test(t, "namespaced", audit.LevelRequest, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted}, "omit RequestReceived", "getPods", "default")
	test(t, "namespaced", audit.LevelRequest, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
	test(t, "cluster", audit.LevelRequest, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted}, "omit RequestReceived", "getPods", "default")
	test(t, "cluster", audit.LevelRequest, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
	test(t, "nonResource", audit.LevelMetadata, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted}, "default", "omit RequestReceived", "getPods")
	test(t, "nonResource", audit.LevelRequest, policyStages, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")
}

// stageEqual returns true if s1 and s2 are super set of each other
func stageEqual(s1, s2 []audit.Stage) bool {
	m1 := make(map[audit.Stage]bool)
	m2 := make(map[audit.Stage]bool)
	for _, s := range s1 {
		m1[s] = true
	}
	for _, s := range s2 {
		m2[s] = true
	}
	if len(m1) != len(m2) {
		return false
	}
	for key, value := range m1 {
		if m2[key] != value {
			return false
		}
	}
	return true
}

func TestUnionStages(t *testing.T) {
	var testCases = []struct {
		s1, s2, exp []audit.Stage
	}{
		{
			[]audit.Stage{},
			[]audit.Stage{},
			[]audit.Stage{},
		},
		{
			[]audit.Stage{audit.StageRequestReceived},
			[]audit.Stage{},
			[]audit.Stage{audit.StageRequestReceived},
		},
		{
			[]audit.Stage{audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived},
		},
		{
			[]audit.Stage{audit.StageRequestReceived},
			[]audit.Stage{audit.StageResponseStarted},
			[]audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted},
		},
		{
			[]audit.Stage{audit.StageRequestReceived, audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived, audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived},
		},
		{
			[]audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted},
			[]audit.Stage{audit.StagePanic, audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StagePanic},
		},
		{
			nil,
			[]audit.Stage{audit.StageRequestReceived},
			[]audit.Stage{audit.StageRequestReceived},
		},
	}

	for _, tc := range testCases {
		result := unionStages(tc.s1, tc.s2)
		assert.Len(t, result, len(tc.exp))
		for _, expStage := range tc.exp {
			ok := false
			for _, resultStage := range result {
				if resultStage == expStage {
					ok = true
					break
				}
			}
			assert.True(t, ok)
		}
	}
}

func TestOmitManagedFields(t *testing.T) {
	// this authorizer.Attributes should match all policy rules
	// specified in this test.
	attributes := &authorizer.AttributesRecord{
		Verb: "get",
	}
	matchingPolicyRule := audit.PolicyRule{
		Level: audit.LevelRequestResponse,
		Verbs: []string{
			attributes.GetVerb(),
		},
	}

	boolPtr := func(v bool) *bool {
		return &v
	}

	tests := []struct {
		name   string
		policy func() audit.Policy
		want   bool
	}{
		{
			name: "global policy default is false, rule does not override",
			policy: func() audit.Policy {
				return audit.Policy{
					OmitManagedFields: false,
					Rules: []audit.PolicyRule{
						*matchingPolicyRule.DeepCopy(),
					},
				}
			},
		},
		{
			name: "global policy default is true, rule does not override",
			policy: func() audit.Policy {
				return audit.Policy{
					OmitManagedFields: true,
					Rules: []audit.PolicyRule{
						*matchingPolicyRule.DeepCopy(),
					},
				}
			},
			want: true,
		},
		{
			name: "global policy default is true, rule overrides to false",
			policy: func() audit.Policy {
				rule := matchingPolicyRule.DeepCopy()
				rule.OmitManagedFields = boolPtr(false)
				return audit.Policy{
					OmitManagedFields: true,
					Rules:             []audit.PolicyRule{*rule},
				}
			},
			want: false,
		},
		{
			name: "global policy default is false, rule overrides to true",
			policy: func() audit.Policy {
				rule := matchingPolicyRule.DeepCopy()
				rule.OmitManagedFields = boolPtr(true)
				return audit.Policy{
					OmitManagedFields: false,
					Rules:             []audit.PolicyRule{*rule},
				}
			},
			want: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluator := &policyRuleEvaluator{
				Policy: test.policy(),
			}

			got := evaluator.EvaluatePolicyRule(attributes)
			if test.want != got.OmitManagedFields {
				t.Errorf("Expected OmitManagedFields to match, want: %t, got: %t", test.want, got.OmitManagedFields)
			}
		})
	}
}
