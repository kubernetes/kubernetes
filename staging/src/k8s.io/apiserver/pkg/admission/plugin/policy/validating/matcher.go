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

package validating

import (
	"k8s.io/api/admissionregistration/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/validatingadmissionpolicy/matching"
)

var _ matching.MatchCriteria = &matchCriteria{}

type matchCriteria struct {
	constraints *v1beta1.MatchResources
}

// GetParsedNamespaceSelector returns the converted LabelSelector which implements labels.Selector
func (m *matchCriteria) GetParsedNamespaceSelector() (labels.Selector, error) {
	return metav1.LabelSelectorAsSelector(m.constraints.NamespaceSelector)
}

// GetParsedObjectSelector returns the converted LabelSelector which implements labels.Selector
func (m *matchCriteria) GetParsedObjectSelector() (labels.Selector, error) {
	return metav1.LabelSelectorAsSelector(m.constraints.ObjectSelector)
}

// GetMatchResources returns the matchConstraints
func (m *matchCriteria) GetMatchResources() v1beta1.MatchResources {
	return *m.constraints
}

type matcher struct {
	Matcher *matching.Matcher
}

func NewMatcher(m *matching.Matcher) Matcher {
	return &matcher{
		Matcher: m,
	}
}

// ValidateInitialization checks if Matcher is initialized.
func (c *matcher) ValidateInitialization() error {
	return c.Matcher.ValidateInitialization()
}

// DefinitionMatches returns whether this ValidatingAdmissionPolicy matches the provided admission resource request
func (c *matcher) DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1beta1.ValidatingAdmissionPolicy) (bool, schema.GroupVersionResource, schema.GroupVersionKind, error) {
	criteria := matchCriteria{constraints: definition.Spec.MatchConstraints}
	return c.Matcher.Matches(a, o, &criteria)
}

// BindingMatches returns whether this ValidatingAdmissionPolicyBinding matches the provided admission resource request
func (c *matcher) BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, binding *v1beta1.ValidatingAdmissionPolicyBinding) (bool, error) {
	if binding.Spec.MatchResources == nil {
		return true, nil
	}
	criteria := matchCriteria{constraints: binding.Spec.MatchResources}
	isMatch, _, _, err := c.Matcher.Matches(a, o, &criteria)
	return isMatch, err
}

func (c *matcher) GetNamespace(name string) (*corev1.Namespace, error) {
	return c.Matcher.GetNamespace(name)
}
