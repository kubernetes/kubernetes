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
	}

	test := func(req string, expected audit.Level, ruleNames ...string) {
		policy := audit.Policy{}
		for _, rule := range ruleNames {
			require.Contains(t, rules, rule)
			policy.Rules = append(policy.Rules, rules[rule])
		}
		require.Contains(t, attrs, req)
		actual := NewChecker(&policy).Level(attrs[req])
		assert.Equal(t, expected, actual, "request:%s rules:%s", req, strings.Join(ruleNames, ","))
	}

	test("namespaced", audit.LevelMetadata, "default")
	test("namespaced", audit.LevelNone, "create")
	test("namespaced", audit.LevelMetadata, "tims")
	test("namespaced", audit.LevelMetadata, "humans")
	test("namespaced", audit.LevelNone, "serviceAccounts")
	test("namespaced", audit.LevelRequestResponse, "getPods")
	test("namespaced", audit.LevelNone, "getClusterRoles")
	test("namespaced", audit.LevelNone, "getLogs")
	test("namespaced", audit.LevelNone, "getMetrics")
	test("namespaced", audit.LevelMetadata, "getMetrics", "serviceAccounts", "default")
	test("namespaced", audit.LevelRequestResponse, "getMetrics", "getPods", "default")

	test("cluster", audit.LevelMetadata, "default")
	test("cluster", audit.LevelNone, "create")
	test("cluster", audit.LevelMetadata, "tims")
	test("cluster", audit.LevelMetadata, "humans")
	test("cluster", audit.LevelNone, "serviceAccounts")
	test("cluster", audit.LevelNone, "getPods")
	test("cluster", audit.LevelRequestResponse, "getClusterRoles")
	test("cluster", audit.LevelNone, "getLogs")
	test("cluster", audit.LevelNone, "getMetrics")
	test("cluster", audit.LevelMetadata, "getMetrics", "serviceAccounts", "default")
	test("cluster", audit.LevelRequestResponse, "getMetrics", "getClusterRoles", "default")

	test("nonResource", audit.LevelMetadata, "default")
	test("nonResource", audit.LevelNone, "create")
	test("nonResource", audit.LevelMetadata, "tims")
	test("nonResource", audit.LevelMetadata, "humans")
	test("nonResource", audit.LevelNone, "serviceAccounts")
	test("nonResource", audit.LevelNone, "getPods")
	test("nonResource", audit.LevelNone, "getClusterRoles")
	test("nonResource", audit.LevelRequestResponse, "getLogs")
	test("nonResource", audit.LevelNone, "getMetrics")
	test("nonResource", audit.LevelMetadata, "getMetrics", "serviceAccounts", "default")
	test("nonResource", audit.LevelRequestResponse, "getLogs", "getClusterRoles", "default")
}
