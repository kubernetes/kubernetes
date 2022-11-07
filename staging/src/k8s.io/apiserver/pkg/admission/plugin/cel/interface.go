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
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/cel"

	"github.com/google/cel-go/common/types/ref"
)

type FailurePolicy string

const (
	Fail   FailurePolicy = "Fail"
	Ignore FailurePolicy = "Ignore"
)

// EvaluatorFunc represents the AND of one or more compiled CEL expression's
// evaluators `params` may be nil if definition does not specify a paramsource
type EvaluatorFunc func(a admission.Attributes, params *unstructured.Unstructured) []PolicyDecision

// ObjectConverter is Dependency Injected into the PolicyDefinition's `Compile`
// function to assist with converting types and values to/from CEL-typed values.
type ObjectConverter interface {
	// DeclForResource looks up the openapi or JSONSchemaProps, structural schema, etc.
	// and compiles it into something that can be used to turn objects into CEL
	// values
	DeclForResource(gvr schema.GroupVersionResource) (*cel.DeclType, error)

	// ValueForObject takes a Kubernetes Object and uses the CEL DeclType
	// to transform it into a CEL value.
	// Object may be a typed native object or an unstructured object
	ValueForObject(value runtime.Object, decl *cel.DeclType) (ref.Val, error)
}

// PolicyDefinition is an interface for internal policy binding type.
// Implemented by mock/testing types, and to be implemented by the public API
// types once they have completed API review.
//
// The interface closely mirrors the format and functionality of the
// PolicyDefinition proposed in the KEP.
type PolicyDefinition interface {
	runtime.Object

	// Matches says whether this policy definition matches the provided admission
	// resource request
	Matches(a admission.Attributes) bool

	Compile(
		// Definition is provided with a converter which may be used by the
		// return evaluator function to convert objects into CEL-typed objects
		objectConverter ObjectConverter,
		// Injected RESTMapper to assist with compilation
		mapper meta.RESTMapper,
	) (EvaluatorFunc, error)

	// GetParamSource returns the GVK for the CRD used as the source of
	// parameters used in the evaluation of instances of this policy
	// May return nil if there is no paramsource for this definition.
	GetParamSource() *schema.GroupVersionKind

	// GetFailurePolicy returns how an object should be treated during an
	// admission when there is a configuration error preventing CEL evaluation
	GetFailurePolicy() FailurePolicy
}

// PolicyBinding is an interface for internal policy binding type. Implemented
// by mock/testing types, and to be implemented by the public API types once
// they have completed API review.
//
// The interface closely mirrors the format and functionality of the
// PolicyBinding proposed in the KEP.
type PolicyBinding interface {
	runtime.Object

	// Matches says whether this policy binding matches the provided admission
	// resource request
	Matches(a admission.Attributes) bool

	// GetTargetDefinition returns the Namespace/Name of Policy Definition used
	// by this binding.
	GetTargetDefinition() (namespace, name string)

	// GetTargetParams returns the Namespace/Name of instance of TargetDefinition's
	// ParamSource to be provided to the CEL expressions of the definition during
	// evaluation.
	// If TargetDefinition has nil ParamSource, this is ignored.
	GetTargetParams() (namespace, name string)
}
