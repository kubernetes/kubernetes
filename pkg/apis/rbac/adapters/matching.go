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

package adapters

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/apis/rbac"
)

func VerbMatches(rule PolicyRule, requestedVerb string) bool {
	for _, ruleVerb := range rule.Verbs() {
		if ruleVerb == rbac.VerbAll {
			return true
		}
		if ruleVerb == requestedVerb {
			return true
		}
	}

	return false
}

func APIGroupMatches(rule PolicyRule, requestedGroup string) bool {
	for _, ruleGroup := range rule.APIGroups() {
		if ruleGroup == rbac.APIGroupAll {
			return true
		}
		if ruleGroup == requestedGroup {
			return true
		}
	}

	return false
}

func ResourceMatches(rule PolicyRule, requestedResource string) bool {
	for _, ruleResource := range rule.Resources() {
		if ruleResource == rbac.ResourceAll {
			return true
		}
		if ruleResource == requestedResource {
			return true
		}
	}

	return false
}

func ResourceNameMatches(rule PolicyRule, requestedName string) bool {
	if len(rule.ResourceNames()) == 0 {
		return true
	}

	for _, ruleName := range rule.ResourceNames() {
		if ruleName == requestedName {
			return true
		}
	}

	return false
}

func NonResourceURLMatches(rule PolicyRule, requestedURL string) bool {
	for _, ruleURL := range rule.NonResourceURLs() {
		if ruleURL == rbac.NonResourceAll {
			return true
		}
		if ruleURL == requestedURL {
			return true
		}
		if strings.HasSuffix(ruleURL, "*") && strings.HasPrefix(requestedURL, strings.TrimRight(ruleURL, "*")) {
			return true
		}
	}

	return false
}

// subjectsStrings returns users, groups, serviceaccounts, unknown for display purposes.
func SubjectsStrings(subjects []Subject) ([]string, []string, []string, []string) {
	users := []string{}
	groups := []string{}
	sas := []string{}
	others := []string{}

	for _, subject := range subjects {
		switch subject.Kind() {
		case rbac.ServiceAccountKind:
			sas = append(sas, fmt.Sprintf("%s/%s", subject.Namespace(), subject.Name()))

		case rbac.UserKind:
			users = append(users, subject.Name())

		case rbac.GroupKind:
			groups = append(groups, subject.Name())

		default:
			others = append(others, fmt.Sprintf("%s/%s/%s", subject.Kind(), subject.Namespace(), subject.Name()))
		}
	}

	return users, groups, sas, others
}
