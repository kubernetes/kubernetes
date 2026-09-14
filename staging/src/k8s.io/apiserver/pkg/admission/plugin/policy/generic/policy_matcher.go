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

package generic

import (
	"context"
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
)

// Matcher is used for matching ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding to attributes
type PolicyMatcher interface {
	admission.InitializationValidator

	// DefinitionMatches says whether this policy definition matches the provided admission
	// resource request
	DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition PolicyAccessor) (bool, schema.GroupVersionResource, schema.GroupVersionKind, error)

	// BindingMatches says whether this policy definition matches the provided admission
	// resource request
	BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, binding BindingAccessor) (bool, error)

	// GetNamespace retrieves the Namespace resource by the given name. The name may be empty, in which case
	// GetNamespace must return nil, NotFound
	GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error)
}

var errNilSelector = "a nil %s selector was passed, please ensure selectors are initialized properly"

type matcher struct {
	Matcher *matching.Matcher
}

func NewPolicyMatcher(m *matching.Matcher) PolicyMatcher {
	return &matcher{
		Matcher: m,
	}
}

// ValidateInitialization checks if Matcher is initialized.
func (c *matcher) ValidateInitialization() error {
	return c.Matcher.ValidateInitialization()
}

// DefinitionMatches returns whether this ValidatingAdmissionPolicy matches the provided admission resource request
func (c *matcher) DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition PolicyAccessor) (bool, schema.GroupVersionResource, schema.GroupVersionKind, error) {
	constraints := definition.GetMatchConstraints()
	if constraints == nil {
		return false, schema.GroupVersionResource{}, schema.GroupVersionKind{}, fmt.Errorf("policy contained no match constraints, a required field")
	}
	if constraints.NamespaceSelector == nil {
		return false, schema.GroupVersionResource{}, schema.GroupVersionKind{}, fmt.Errorf(errNilSelector, "namespace")
	}
	if constraints.ObjectSelector == nil {
		return false, schema.GroupVersionResource{}, schema.GroupVersionKind{}, fmt.Errorf(errNilSelector, "object")
	}

	criteria := matchCriteria{constraints: constraints}
	return c.Matcher.Matches(a, o, &criteria)
}

// BindingMatches returns whether this ValidatingAdmissionPolicyBinding matches the provided admission resource request
func (c *matcher) BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, binding BindingAccessor) (bool, error) {
	matchResources := binding.GetMatchResources()
	if matchResources == nil {
		return true, nil
	}
	if matchResources.NamespaceSelector == nil {
		return false, fmt.Errorf(errNilSelector, "namespace")
	}
	if matchResources.ObjectSelector == nil {
		return false, fmt.Errorf(errNilSelector, "object")
	}

	criteria := matchCriteria{constraints: matchResources}
	isMatch, _, _, err := c.Matcher.Matches(a, o, &criteria)
	return isMatch, err
}

func (c *matcher) GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error) {
	return c.Matcher.GetNamespace(ctx, name)
}

var _ matching.MatchCriteria = &matchCriteria{}

type matchCriteria struct {
	constraints *admissionregistrationv1.MatchResources
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
func (m *matchCriteria) GetMatchResources() admissionregistrationv1.MatchResources {
	return *m.constraints
}
