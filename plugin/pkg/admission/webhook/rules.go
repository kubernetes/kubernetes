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

import (
	"strings"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1"
)

type RuleMatcher struct {
	Rule v1alpha1.RuleWithOperations
	Attr admission.Attributes
}

func (r *RuleMatcher) Matches() bool {
	return r.operation() &&
		r.group() &&
		r.version() &&
		r.resource()
}

func exactOrWildcard(items []string, requested string) bool {
	for _, item := range items {
		if item == "*" {
			return true
		}
		if item == requested {
			return true
		}
	}

	return false
}

func (r *RuleMatcher) group() bool {
	return exactOrWildcard(r.Rule.APIGroups, r.Attr.GetResource().Group)
}

func (r *RuleMatcher) version() bool {
	return exactOrWildcard(r.Rule.APIVersions, r.Attr.GetResource().Version)
}

func (r *RuleMatcher) operation() bool {
	attrOp := r.Attr.GetOperation()
	for _, op := range r.Rule.Operations {
		if op == v1alpha1.OperationAll {
			return true
		}
		// The constants are the same such that this is a valid cast (and this
		// is tested).
		if op == v1alpha1.OperationType(attrOp) {
			return true
		}
	}
	return false
}

func splitResource(resSub string) (res, sub string) {
	parts := strings.SplitN(resSub, "/", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return parts[0], ""
}

func (r *RuleMatcher) resource() bool {
	opRes, opSub := r.Attr.GetResource().Resource, r.Attr.GetSubresource()
	for _, res := range r.Rule.Resources {
		res, sub := splitResource(res)
		resMatch := res == "*" || res == opRes
		subMatch := sub == "*" || sub == opSub
		if resMatch && subMatch {
			return true
		}
	}
	return false
}
