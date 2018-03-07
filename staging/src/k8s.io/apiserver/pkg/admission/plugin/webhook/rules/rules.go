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

package rules

import (
	"strings"

	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apiserver/pkg/admission"
)

// Matcher determines if the Attr matches the Rule.
type Matcher struct {
	Rule v1beta1.RuleWithOperations
	Attr admission.Attributes
}

// Matches returns if the Attr matches the Rule.
func (r *Matcher) Matches() bool {
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

func (r *Matcher) group() bool {
	return exactOrWildcard(r.Rule.APIGroups, r.Attr.GetResource().Group)
}

func (r *Matcher) version() bool {
	return exactOrWildcard(r.Rule.APIVersions, r.Attr.GetResource().Version)
}

func (r *Matcher) operation() bool {
	attrOp := r.Attr.GetOperation()
	for _, op := range r.Rule.Operations {
		if op == v1beta1.OperationAll {
			return true
		}
		// The constants are the same such that this is a valid cast (and this
		// is tested).
		if op == v1beta1.OperationType(attrOp) {
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

func (r *Matcher) resource() bool {
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

// IsWebhookConfigurationResource determines if an admission.Attributes object is describing
// the admission of a ValidatingWebhookConfiguration or a MutatingWebhookConfiguration
func IsWebhookConfigurationResource(attr admission.Attributes) bool {
	gvk := attr.GetKind()
	if gvk.Group == "admissionregistration.k8s.io" {
		if gvk.Kind == "ValidatingWebhookConfiguration" || gvk.Kind == "MutatingWebhookConfiguration" {
			return true
		}
	}
	return false
}
