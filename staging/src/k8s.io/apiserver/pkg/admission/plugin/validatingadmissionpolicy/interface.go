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
	"context"

	celgo "github.com/google/cel-go/cel"

	"k8s.io/api/admissionregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
)

var _ cel.ExpressionAccessor = &ValidationCondition{}

// ValidationCondition contains the inputs needed to compile, evaluate and validate a cel expression
type ValidationCondition struct {
	Expression string
	Message    string
	Reason     *metav1.StatusReason
}

func (v *ValidationCondition) GetExpression() string {
	return v.Expression
}

func (v *ValidationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}

// AuditAnnotationCondition contains the inputs needed to compile, evaluate and publish a cel audit annotation
type AuditAnnotationCondition struct {
	Key             string
	ValueExpression string
}

func (v *AuditAnnotationCondition) GetExpression() string {
	return v.ValueExpression
}

func (v *AuditAnnotationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.StringType, celgo.NullType}
}

// Matcher is used for matching ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding to attributes
type Matcher interface {
	admission.InitializationValidator

	// DefinitionMatches says whether this policy definition matches the provided admission
	// resource request
	DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1alpha1.ValidatingAdmissionPolicy) (bool, schema.GroupVersionKind, error)

	// BindingMatches says whether this policy definition matches the provided admission
	// resource request
	BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1alpha1.ValidatingAdmissionPolicyBinding) (bool, error)
}

// ValidateResult defines the result of a Validator.Validate operation.
type ValidateResult struct {
	// Decisions specifies the outcome of the validation as well as the details about the decision.
	Decisions []PolicyDecision
	// AuditAnnotations specifies the audit annotations that should be recorded for the validation.
	AuditAnnotations []PolicyAuditAnnotation
}

// Validator is contains logic for converting ValidationEvaluation to PolicyDecisions
type Validator interface {
	// Validate is used to take cel evaluations and convert into decisions
	// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
	Validate(ctx context.Context, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, runtimeCELCostBudget int64) ValidateResult
}
