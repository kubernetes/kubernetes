/*
Copyright 2019 The Kubernetes Authors.

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

package flowcontrol

import (
	"strings"

	flowcontrol "k8s.io/api/flowcontrol/v1beta3"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// Tests whether a given request and FlowSchema match.  Nobody mutates
// either input.
func matchesFlowSchema(digest RequestDigest, flowSchema *flowcontrol.FlowSchema) bool {
	for _, policyRule := range flowSchema.Spec.Rules {
		if matchesPolicyRule(digest, &policyRule) {
			return true
		}
	}
	return false
}

func matchesPolicyRule(digest RequestDigest, policyRule *flowcontrol.PolicyRulesWithSubjects) bool {
	if !matchesASubject(digest.User, policyRule.Subjects) {
		return false
	}
	if digest.RequestInfo.IsResourceRequest {
		return matchesAResourceRule(digest.RequestInfo, policyRule.ResourceRules)
	}
	return matchesANonResourceRule(digest.RequestInfo, policyRule.NonResourceRules)
}

func matchesASubject(user user.Info, subjects []flowcontrol.Subject) bool {
	for _, subject := range subjects {
		if matchesSubject(user, subject) {
			return true
		}
	}
	return false
}

func matchesSubject(user user.Info, subject flowcontrol.Subject) bool {
	switch subject.Kind {
	case flowcontrol.SubjectKindUser:
		return subject.User != nil && (subject.User.Name == flowcontrol.NameAll || subject.User.Name == user.GetName())
	case flowcontrol.SubjectKindGroup:
		if subject.Group == nil {
			return false
		}
		seek := subject.Group.Name
		if seek == "*" {
			return true
		}
		for _, userGroup := range user.GetGroups() {
			if userGroup == seek {
				return true
			}
		}
		return false
	case flowcontrol.SubjectKindServiceAccount:
		if subject.ServiceAccount == nil {
			return false
		}
		if subject.ServiceAccount.Name == flowcontrol.NameAll {
			return serviceAccountMatchesNamespace(subject.ServiceAccount.Namespace, user.GetName())
		}
		return serviceaccount.MatchesUsername(subject.ServiceAccount.Namespace, subject.ServiceAccount.Name, user.GetName())
	default:
		return false
	}
}

// serviceAccountMatchesNamespace checks whether the provided service account username matches the namespace, without
// allocating. Use this when checking a service account namespace against a known string.
// This is copied from `k8s.io/apiserver/pkg/authentication/serviceaccount::MatchesUsername` and simplified to not check the name part.
func serviceAccountMatchesNamespace(namespace string, username string) bool {
	const (
		ServiceAccountUsernamePrefix    = "system:serviceaccount:"
		ServiceAccountUsernameSeparator = ":"
	)
	if !strings.HasPrefix(username, ServiceAccountUsernamePrefix) {
		return false
	}
	username = username[len(ServiceAccountUsernamePrefix):]

	if !strings.HasPrefix(username, namespace) {
		return false
	}
	username = username[len(namespace):]

	return strings.HasPrefix(username, ServiceAccountUsernameSeparator)
}

func matchesAResourceRule(ri *request.RequestInfo, rules []flowcontrol.ResourcePolicyRule) bool {
	for _, rr := range rules {
		if matchesResourcePolicyRule(ri, rr) {
			return true
		}
	}
	return false
}

func matchesResourcePolicyRule(ri *request.RequestInfo, policyRule flowcontrol.ResourcePolicyRule) bool {
	if !matchPolicyRuleVerb(policyRule.Verbs, ri.Verb) {
		return false
	}
	if !matchPolicyRuleResource(policyRule.Resources, ri.Resource, ri.Subresource) {
		return false
	}
	if !matchPolicyRuleAPIGroup(policyRule.APIGroups, ri.APIGroup) {
		return false
	}
	if len(ri.Namespace) == 0 {
		return policyRule.ClusterScope
	}
	return containsString(ri.Namespace, policyRule.Namespaces, flowcontrol.NamespaceEvery)
}

func matchesANonResourceRule(ri *request.RequestInfo, rules []flowcontrol.NonResourcePolicyRule) bool {
	for _, rr := range rules {
		if matchesNonResourcePolicyRule(ri, rr) {
			return true
		}
	}
	return false
}

func matchesNonResourcePolicyRule(ri *request.RequestInfo, policyRule flowcontrol.NonResourcePolicyRule) bool {
	if !matchPolicyRuleVerb(policyRule.Verbs, ri.Verb) {
		return false
	}
	return matchPolicyRuleNonResourceURL(policyRule.NonResourceURLs, ri.Path)
}

func matchPolicyRuleVerb(policyRuleVerbs []string, requestVerb string) bool {
	return containsString(requestVerb, policyRuleVerbs, flowcontrol.VerbAll)
}

func matchPolicyRuleNonResourceURL(policyRuleRequestURLs []string, requestPath string) bool {
	for _, rulePath := range policyRuleRequestURLs {
		if rulePath == flowcontrol.NonResourceAll || rulePath == requestPath {
			return true
		}
		rulePrefix := strings.TrimSuffix(rulePath, "*")
		if !strings.HasSuffix(rulePrefix, "/") {
			rulePrefix = rulePrefix + "/"
		}
		if strings.HasPrefix(requestPath, rulePrefix) {
			return true
		}
	}
	return false
}

func matchPolicyRuleAPIGroup(policyRuleAPIGroups []string, requestAPIGroup string) bool {
	return containsString(requestAPIGroup, policyRuleAPIGroups, flowcontrol.APIGroupAll)
}

func rsJoin(requestResource, requestSubresource string) string {
	seekString := requestResource
	if requestSubresource != "" {
		seekString = requestResource + "/" + requestSubresource
	}
	return seekString
}

func matchPolicyRuleResource(policyRuleRequestResources []string, requestResource, requestSubresource string) bool {
	return containsString(rsJoin(requestResource, requestSubresource), policyRuleRequestResources, flowcontrol.ResourceAll)
}

// containsString returns true if either `x` or `wildcard` is in
// `list`.  The wildcard is not a pattern to match against `x`; rather
// the presence of the wildcard in the list is the caller's way of
// saying that all values of `x` should match the list.  This function
// assumes that if `wildcard` is in `list` then it is the only member
// of the list, which is enforced by validation.
func containsString(x string, list []string, wildcard string) bool {
	if len(list) == 1 && list[0] == wildcard {
		return true
	}
	for _, y := range list {
		if x == y {
			return true
		}
	}
	return false
}
