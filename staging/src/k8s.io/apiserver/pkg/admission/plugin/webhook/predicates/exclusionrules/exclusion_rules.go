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

package exclusionrules

import (
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

// Matcher determines if the Attr matches the Rule.
type Matcher struct {
	ExclusionRule webhookadmission.ExclusionRule
	Attr          admission.Attributes
}

// Matches returns if the Attr matches the Rule.
func (r *Matcher) Matches() bool {
	return r.name() &&
		r.namespace() &&
		r.group() &&
		r.version() &&
		r.kind()
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
	return exactOrWildcard(r.ExclusionRule.APIGroups, r.Attr.GetResource().Group)
}

func (r *Matcher) name() bool {
	return r.ExclusionRule.Name == r.Attr.GetName()
}

func (r *Matcher) namespace() bool {
	return r.ExclusionRule.Namespace == r.Attr.GetNamespace()
}

func (r *Matcher) version() bool {
	return exactOrWildcard(r.ExclusionRule.APIVersions, r.Attr.GetResource().Version)
}

func (r *Matcher) kind() bool {
	return r.ExclusionRule.Kind == r.Attr.GetKind().Kind
}
