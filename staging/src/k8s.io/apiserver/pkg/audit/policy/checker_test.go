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

func TestChecker(t *testing.T) {
	tim := &user.DefaultInfo{
		Name:   "tim@k8s.io",
		Groups: []string{"humans", "developers"},
	}
	attrs := map[string]authorizer.Attributes{
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
	}

	rules := map[string]audit.PolicyRule{
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

	test := func(req string, expLevel audit.Level, expOmitStages []audit.Stage, ruleNames ...string) {
		policy := audit.Policy{}
		for _, rule := range ruleNames {
			require.Contains(t, rules, rule)
			policy.Rules = append(policy.Rules, rules[rule])
		}
		require.Contains(t, attrs, req)
		actualLevel, actualOmitStages := NewChecker(&policy).LevelAndStages(attrs[req])
		assert.Equal(t, expLevel, actualLevel, "request:%s rules:%s", req, strings.Join(ruleNames, ","))
		assert.Equal(t, expOmitStages, actualOmitStages, "request:%s rules:%s", req, strings.Join(ruleNames, ","))
	}

	test("namespaced", audit.LevelMetadata, nil, "default")
	test("namespaced", audit.LevelNone, nil, "create")
	test("namespaced", audit.LevelMetadata, nil, "tims")
	test("namespaced", audit.LevelMetadata, nil, "humans")
	test("namespaced", audit.LevelNone, nil, "serviceAccounts")
	test("namespaced", audit.LevelRequestResponse, nil, "getPods")
	test("namespaced", audit.LevelNone, nil, "getClusterRoles")
	test("namespaced", audit.LevelNone, nil, "getLogs")
	test("namespaced", audit.LevelNone, nil, "getMetrics")
	test("namespaced", audit.LevelMetadata, nil, "getMetrics", "serviceAccounts", "default")
	test("namespaced", audit.LevelRequestResponse, nil, "getMetrics", "getPods", "default")
	test("namespaced", audit.LevelRequestResponse, nil, "getPodLogs", "getPods")
	test("namespaced", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test("namespaced", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")

	test("cluster", audit.LevelMetadata, nil, "default")
	test("cluster", audit.LevelNone, nil, "create")
	test("cluster", audit.LevelMetadata, nil, "tims")
	test("cluster", audit.LevelMetadata, nil, "humans")
	test("cluster", audit.LevelNone, nil, "serviceAccounts")
	test("cluster", audit.LevelNone, nil, "getPods")
	test("cluster", audit.LevelRequestResponse, nil, "getClusterRoles")
	test("cluster", audit.LevelRequest, nil, "clusterRoleEdit", "getClusterRoles")
	test("cluster", audit.LevelNone, nil, "getLogs")
	test("cluster", audit.LevelNone, nil, "getMetrics")
	test("cluster", audit.LevelMetadata, nil, "getMetrics", "serviceAccounts", "default")
	test("cluster", audit.LevelRequestResponse, nil, "getMetrics", "getClusterRoles", "default")
	test("cluster", audit.LevelNone, nil, "getPodLogs", "getPods")
	test("cluster", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test("cluster", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")

	test("nonResource", audit.LevelMetadata, nil, "default")
	test("nonResource", audit.LevelNone, nil, "create")
	test("nonResource", audit.LevelMetadata, nil, "tims")
	test("nonResource", audit.LevelMetadata, nil, "humans")
	test("nonResource", audit.LevelNone, nil, "serviceAccounts")
	test("nonResource", audit.LevelNone, nil, "getPods")
	test("nonResource", audit.LevelNone, nil, "getClusterRoles")
	test("nonResource", audit.LevelRequestResponse, nil, "getLogs")
	test("nonResource", audit.LevelNone, nil, "getMetrics")
	test("nonResource", audit.LevelMetadata, nil, "getMetrics", "serviceAccounts", "default")
	test("nonResource", audit.LevelRequestResponse, nil, "getLogs", "getClusterRoles", "default")
	test("nonResource", audit.LevelNone, nil, "getPodLogs", "getPods")
	test("nonResource", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived}, "omit RequestReceived", "getPods", "default")
	test("nonResource", audit.LevelRequest, []audit.Stage{audit.StageRequestReceived, audit.StageResponseStarted, audit.StageResponseComplete}, "only audit panic", "getPods", "default")

	test("subresource", audit.LevelRequest, nil, "getPodLogs", "getPods")
	test("subresource", audit.LevelRequest, nil, "getPods", "getPodLogs")
}
