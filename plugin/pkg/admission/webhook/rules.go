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

// Package webhook checks a webhook for configured operation admission
package webhook

import "k8s.io/apiserver/pkg/admission"

func indexOf(items []string, requested string) int {
	for i, item := range items {
		if item == requested {
			return i
		}
	}

	return -1
}

// APIGroupMatches returns if the admission.Attributes matches the rule's API groups
func APIGroupMatches(rule Rule, attr admission.Attributes) bool {
	if len(rule.APIGroups) == 0 {
		return true
	}

	return indexOf(rule.APIGroups, attr.GetResource().Group) > -1
}

// Matches returns if the admission.Attributes matches the rule
func Matches(rule Rule, attr admission.Attributes) bool {
	return APIGroupMatches(rule, attr) &&
		NamespaceMatches(rule, attr) &&
		OperationMatches(rule, attr) &&
		ResourceMatches(rule, attr) &&
		ResourceNamesMatches(rule, attr)
}

// OperationMatches returns if the admission.Attributes matches the rule's operation
func OperationMatches(rule Rule, attr admission.Attributes) bool {
	if len(rule.Operations) == 0 {
		return true
	}

	aOp := attr.GetOperation()

	for _, rOp := range rule.Operations {
		if aOp == rOp {
			return true
		}
	}

	return false
}

// NamespaceMatches returns if the admission.Attributes matches the rule's namespaces
func NamespaceMatches(rule Rule, attr admission.Attributes) bool {
	if len(rule.Namespaces) == 0 {
		return true
	}

	return indexOf(rule.Namespaces, attr.GetNamespace()) > -1
}

// ResourceMatches returns if the admission.Attributes matches the rule's resource (and optional subresource)
func ResourceMatches(rule Rule, attr admission.Attributes) bool {
	if len(rule.Resources) == 0 {
		return true
	}

	resource := attr.GetResource().Resource

	if len(attr.GetSubresource()) > 0 {
		resource = attr.GetResource().Resource + "/" + attr.GetSubresource()
	}

	return indexOf(rule.Resources, resource) > -1
}

// ResourceNamesMatches returns if the admission.Attributes matches the rule's resource names
func ResourceNamesMatches(rule Rule, attr admission.Attributes) bool {
	if len(rule.ResourceNames) == 0 {
		return true
	}

	return indexOf(rule.ResourceNames, attr.GetName()) > -1
}
