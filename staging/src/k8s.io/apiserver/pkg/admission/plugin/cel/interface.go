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

package cel

import (
	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apiserver/pkg/admission"
)

// EvaluatorFunc represents the AND of one or more compiled CEL expression's
// evaluators `params` may be nil if definition does not specify a paramsource
type ValidatorFunc func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error)

func (f ValidatorFunc) Validate(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
	return f(a, params)
}

type Validator interface {
	Validate(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error)
}

// ValidatorCompiler is Dependency Injected into the PolicyDefinition's `Compile`
// function to assist with converting types and values to/from CEL-typed values.
type ValidatorCompiler interface {
	// Matches says whether this policy definition matches the provided admission
	// resource request
	DefinitionMatches(definition *v1alpha1.ValidatingAdmissionPolicy, a admission.Attributes) bool

	// Matches says whether this policy definition matches the provided admission
	// resource request
	BindingMatches(definition *v1alpha1.ValidatingAdmissionPolicyBinding, a admission.Attributes) bool

	Compile(
		policy *v1alpha1.ValidatingAdmissionPolicy,
		// Injected RESTMapper to assist with compilation
		mapper meta.RESTMapper,
	) (Validator, error)
}
