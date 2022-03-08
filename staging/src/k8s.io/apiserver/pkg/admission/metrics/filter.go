/*
Copyright 2022 The Kubernetes Authors.

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

package metrics

import (
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

type LabelCardinalityFilter []webhookadmission.Rule

func (f LabelCardinalityFilter) matches(gvr schema.GroupVersionResource, subresource string) bool {
	for _, rule := range f {
		if matchesGroup(rule, gvr) && matchesVersion(rule, gvr) && matchesResource(rule, gvr, subresource) {
			return true
		}
	}
	return false
}

func matchesGroup(rule webhookadmission.Rule, gvr schema.GroupVersionResource) bool {
	return matchesPatterns(rule.Groups, gvr.Group)
}

func matchesVersion(rule webhookadmission.Rule, gvr schema.GroupVersionResource) bool {
	return matchesPatterns(rule.Versions, gvr.Version)
}

func matchesResource(rule webhookadmission.Rule, gvr schema.GroupVersionResource, subresource string) bool {
	for _, resourceExpression := range rule.Resources {
		var expectedResource, expectedSubresource string
		rs := strings.Split(resourceExpression, "/")
		expectedResource = rs[0]
		if len(rs) > 1 {
			expectedSubresource = rs[1]
		}
		matchedResource := expectedResource == "*" || expectedResource == gvr.Resource
		matchedSubresource :=
			(len(expectedSubresource) == 0 && len(subresource) == 0) || // if only matching resource and there is no subresource
				(expectedSubresource == "*" && len(subresource) > 0) || // if wildcard and there is a subresource
				expectedSubresource == subresource // exact subresource match
		if matchedResource && matchedSubresource {
			return true
		}
	}
	return false
}

func matchesPatterns(patterns []string, s string) bool {
	for _, pattern := range patterns {
		if pattern == "*" || pattern == s {
			return true
		}
	}
	return false
}
