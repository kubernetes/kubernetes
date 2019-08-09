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
	rmtypesv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
)

func matchesFlowSchema(digest RequestDigest, flowSchema *rmtypesv1alpha1.FlowSchema) bool {
	for _, policyRule := range flowSchema.Spec.Rules {
		subjectMatches := false
		for _, subject := range policyRule.Subjects {
			if matchesSubject(digest, subject) {
				subjectMatches = true
				break
			}
		}
		if !subjectMatches {
			continue
		}

		if matchesPolicyRule(digest, policyRule.Rule) {
			return true
		}
	}
	return false
}

func matchesSubject(digest RequestDigest, subject rmtypesv1alpha1.Subject) bool {
	user := digest.User
	switch subject.Kind {
	case rmtypesv1alpha1.UserKind:
		return user.GetName() == subject.Name
	case rmtypesv1alpha1.GroupKind:
		return containsString(subject.Name, user.GetGroups(), "")
	case rmtypesv1alpha1.ServiceAccountKind:
		return serviceaccount.MatchesUsername(subject.Namespace, subject.Name, user.GetName())
	default:
		return false
	}
}

func matchesPolicyRule(digest RequestDigest, policyRule rmtypesv1alpha1.PolicyRule) bool {
	if !matchPolicyRuleVerb(policyRule.Verbs, digest.RequestInfo.Verb) {
		return false
	}
	if !digest.RequestInfo.IsResourceRequest {
		return matchPolicyRuleNonResourceURL(policyRule.NonResourceURLs, digest.RequestInfo.Path)
	}
	if !matchPolicyRuleResource(policyRule.Resources, digest.RequestInfo.Resource) {
		return false
	}
	if !matchPolicyRuleAPIGroup(policyRule.APIGroups, digest.RequestInfo.APIGroup) {
		return false
	}
	return true
}

func matchPolicyRuleVerb(policyRuleVerbs []string, requestVerb string) bool {
	return containsString(requestVerb, policyRuleVerbs, rmtypesv1alpha1.VerbAll)
}

func matchPolicyRuleNonResourceURL(policyRuleRequestURLs []string, requestPath string) bool {
	return containsString(requestPath, policyRuleRequestURLs, rmtypesv1alpha1.NonResourceAll)
}

func matchPolicyRuleAPIGroup(policyRuleAPIGroups []string, requestAPIGroup string) bool {
	return containsString(requestAPIGroup, policyRuleAPIGroups, rmtypesv1alpha1.APIGroupAll)
}

func matchPolicyRuleResource(policyRuleRequestResources []string, requestResource string) bool {
	return containsString(requestResource, policyRuleRequestResources, rmtypesv1alpha1.ResourceAll)
}

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
