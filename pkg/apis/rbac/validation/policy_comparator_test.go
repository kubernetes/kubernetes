/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/apis/rbac"
)

type escalationTest struct {
	ownerRules   []rbac.PolicyRule
	servantRules []rbac.PolicyRule

	expectedCovered        bool
	expectedUncoveredRules []rbac.PolicyRule
}

func TestCoversExactMatch(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversMultipleRulesCoveringSingleRule(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"delete"}, Resources: []string{"deployments"}},
			{APIGroups: []string{"v1"}, Verbs: []string{"delete"}, Resources: []string{"builds"}},
			{APIGroups: []string{"v1"}, Verbs: []string{"update"}, Resources: []string{"builds", "deployments"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"delete", "update"}, Resources: []string{"builds", "deployments"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)

}

func TestCoversMultipleAPIGroupsCoveringSingleRule(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"group1"}, Verbs: []string{"delete"}, Resources: []string{"deployments"}},
			{APIGroups: []string{"group1"}, Verbs: []string{"delete"}, Resources: []string{"builds"}},
			{APIGroups: []string{"group1"}, Verbs: []string{"update"}, Resources: []string{"builds", "deployments"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"delete"}, Resources: []string{"deployments"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"delete"}, Resources: []string{"builds"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"update"}, Resources: []string{"builds", "deployments"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"group1", "group2"}, Verbs: []string{"delete", "update"}, Resources: []string{"builds", "deployments"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)

}

func TestCoversSingleAPIGroupsCoveringMultiple(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"group1", "group2"}, Verbs: []string{"delete", "update"}, Resources: []string{"builds", "deployments"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"group1"}, Verbs: []string{"delete"}, Resources: []string{"deployments"}},
			{APIGroups: []string{"group1"}, Verbs: []string{"delete"}, Resources: []string{"builds"}},
			{APIGroups: []string{"group1"}, Verbs: []string{"update"}, Resources: []string{"builds", "deployments"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"delete"}, Resources: []string{"deployments"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"delete"}, Resources: []string{"builds"}},
			{APIGroups: []string{"group2"}, Verbs: []string{"update"}, Resources: []string{"builds", "deployments"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)

}

func TestCoversMultipleRulesMissingSingleVerbResourceCombination(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"delete", "update"}, Resources: []string{"builds", "deployments"}},
			{APIGroups: []string{"v1"}, Verbs: []string{"delete"}, Resources: []string{"pods"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"delete", "update"}, Resources: []string{"builds", "deployments", "pods"}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"update"}, Resources: []string{"pods"}},
		},
	}.test(t)
}

func TestCoversAPIGroupStarCoveringMultiple(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"*"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"group1", "group2"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversEnumerationNotCoveringAPIGroupStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"dummy-group"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"*"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{APIGroups: []string{"*"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},
	}.test(t)
}

func TestCoversAPIGroupStarCoveringStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"*"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"*"}, Verbs: []string{"get"}, Resources: []string{"roles"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversVerbStarCoveringMultiple(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"*"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"watch", "list"}, Resources: []string{"roles"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversEnumerationNotCoveringVerbStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get", "list", "watch", "create", "update", "delete", "exec"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"*"}, Resources: []string{"roles"}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"*"}, Resources: []string{"roles"}},
		},
	}.test(t)
}

func TestCoversVerbStarCoveringStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"*"}, Resources: []string{"roles"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"*"}, Resources: []string{"roles"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversResourceStarCoveringMultiple(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"resourcegroup:deployments"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversEnumerationNotCoveringResourceStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"roles", "resourcegroup:deployments"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"*"}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"*"}},
		},
	}.test(t)
}

func TestCoversResourceStarCoveringStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"*"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversResourceNameEmptyCoveringMultiple(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"pods"}, ResourceNames: []string{}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"pods"}, ResourceNames: []string{"foo", "bar"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversEnumerationNotCoveringResourceNameEmpty(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"pods"}, ResourceNames: []string{"foo", "bar"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"pods"}, ResourceNames: []string{}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"pods"}},
		},
	}.test(t)
}

func TestCoversNonResourceURLs(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis"}, Verbs: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis"}, Verbs: []string{"*"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversNonResourceURLsStar(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"*"}, Verbs: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis", "/apis/v1", "/"}, Verbs: []string{"*"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversNonResourceURLsStarAfterPrefixDoesntCover(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis/*"}, Verbs: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis", "/apis/v1"}, Verbs: []string{"get"}},
		},

		expectedCovered: false,
		expectedUncoveredRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis"}, Verbs: []string{"get"}},
		},
	}.test(t)
}

func TestCoversNonResourceURLsStarAfterPrefix(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis/*"}, Verbs: []string{"*"}},
		},
		servantRules: []rbac.PolicyRule{
			{NonResourceURLs: []string{"/apis/v1/foo", "/apis/v1"}, Verbs: []string{"get"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversNonResourceURLsWithOtherFields(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}, NonResourceURLs: []string{"/apis"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}, NonResourceURLs: []string{"/apis"}},
		},

		expectedCovered:        true,
		expectedUncoveredRules: []rbac.PolicyRule{},
	}.test(t)
}

func TestCoversNonResourceURLsWithOtherFieldsFailure(t *testing.T) {
	escalationTest{
		ownerRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}},
		},
		servantRules: []rbac.PolicyRule{
			{APIGroups: []string{"v1"}, Verbs: []string{"get"}, Resources: []string{"builds"}, NonResourceURLs: []string{"/apis"}},
		},

		expectedCovered:        false,
		expectedUncoveredRules: []rbac.PolicyRule{{NonResourceURLs: []string{"/apis"}, Verbs: []string{"get"}}},
	}.test(t)
}

func (test escalationTest) test(t *testing.T) {
	actualCovered, actualUncoveredRules := Covers(test.ownerRules, test.servantRules)

	if actualCovered != test.expectedCovered {
		t.Errorf("expected %v, but got %v", test.expectedCovered, actualCovered)
	}

	if !rulesMatch(test.expectedUncoveredRules, actualUncoveredRules) {
		t.Errorf("expected %v, but got %v", test.expectedUncoveredRules, actualUncoveredRules)
	}
}

func rulesMatch(expectedRules, actualRules []rbac.PolicyRule) bool {
	if len(expectedRules) != len(actualRules) {
		return false
	}

	for _, expectedRule := range expectedRules {
		found := false
		for _, actualRule := range actualRules {
			if reflect.DeepEqual(expectedRule, actualRule) {
				found = true
				break
			}
		}

		if !found {
			return false
		}
	}

	return true
}
