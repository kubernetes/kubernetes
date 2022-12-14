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

package validatingadmissionpolicy

import (
	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

// Validator defines the func used to validate the cel expressions
// matchKind provides the GroupVersionKind that the object should be
// validated by CEL expressions as.
type Validator interface {
	Validate(a admission.Attributes, o admission.ObjectInterfaces, versionedParams runtime.Object, matchKind schema.GroupVersionKind) ([]policyDecision, error)
}

// ValidatorCompiler is Dependency Injected into the PolicyDefinition's `Compile`
// function to assist with converting types and values to/from CEL-typed values.
type ValidatorCompiler interface {
	admission.InitializationValidator

	// Matches says whether this policy definition matches the provided admission
	// resource request
	DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1alpha1.ValidatingAdmissionPolicy) (bool, schema.GroupVersionKind, error)

	// Matches says whether this policy definition matches the provided admission
	// resource request
	BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1alpha1.ValidatingAdmissionPolicyBinding) (bool, error)

	// Compile is used for the cel expression compilation
	Compile(
		policy *v1alpha1.ValidatingAdmissionPolicy,
	) Validator
}
