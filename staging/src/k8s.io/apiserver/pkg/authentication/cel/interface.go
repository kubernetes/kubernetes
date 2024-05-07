/*
Copyright 2023 The Kubernetes Authors.

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

// Package cel contains the CEL related interfaces and structs for authentication.
package cel

import (
	"context"

	celgo "github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// ExpressionAccessor is an interface that provides access to a CEL expression.
type ExpressionAccessor interface {
	GetExpression() string
	ReturnTypes() []*celgo.Type
}

// CompilationResult represents a compiled validations expression.
type CompilationResult struct {
	Program            celgo.Program
	ExpressionAccessor ExpressionAccessor
}

// EvaluationResult contains the minimal required fields and metadata of a cel evaluation
type EvaluationResult struct {
	EvalResult         ref.Val
	ExpressionAccessor ExpressionAccessor
}

// Compiler provides a CEL expression compiler configured with the desired authentication related CEL variables.
type Compiler interface {
	CompileClaimsExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error)
	CompileUserExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error)
}

// ClaimsMapper provides a CEL expression mapper configured with the claims CEL variable.
type ClaimsMapper interface {
	// EvalClaimMapping evaluates the given claim mapping expression and returns a EvaluationResult.
	// This is used for username, groups and uid claim mapping that contains a single expression.
	EvalClaimMapping(ctx context.Context, claims *unstructured.Unstructured) (EvaluationResult, error)
	// EvalClaimMappings evaluates the given expressions and returns a list of EvaluationResult.
	// This is used for extra claim mapping and claim validation that contains a list of expressions.
	EvalClaimMappings(ctx context.Context, claims *unstructured.Unstructured) ([]EvaluationResult, error)
}

// UserMapper provides a CEL expression mapper configured with the user CEL variable.
type UserMapper interface {
	// EvalUser evaluates the given user expressions and returns a list of EvaluationResult.
	// This is used for user validation that contains a list of expressions.
	EvalUser(ctx context.Context, userInfo *unstructured.Unstructured) ([]EvaluationResult, error)
}

var _ ExpressionAccessor = &ClaimMappingExpression{}

// ClaimMappingExpression is a CEL expression that maps a claim.
type ClaimMappingExpression struct {
	Expression string
}

// GetExpression returns the CEL expression.
func (v *ClaimMappingExpression) GetExpression() string {
	return v.Expression
}

// ReturnTypes returns the CEL expression return types.
func (v *ClaimMappingExpression) ReturnTypes() []*celgo.Type {
	// return types is only used for validation. The claims variable that's available
	// to the claim mapping expressions is a map[string]interface{}, so we can't
	// really know what the return type is during compilation. Strict type checking
	// is done during evaluation.
	return []*celgo.Type{celgo.AnyType}
}

var _ ExpressionAccessor = &ClaimValidationCondition{}

// ClaimValidationCondition is a CEL expression that validates a claim.
type ClaimValidationCondition struct {
	Expression string
	Message    string
}

// GetExpression returns the CEL expression.
func (v *ClaimValidationCondition) GetExpression() string {
	return v.Expression
}

// ReturnTypes returns the CEL expression return types.
func (v *ClaimValidationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}

var _ ExpressionAccessor = &ExtraMappingExpression{}

// ExtraMappingExpression is a CEL expression that maps an extra to a list of values.
type ExtraMappingExpression struct {
	Key        string
	Expression string
}

// GetExpression returns the CEL expression.
func (v *ExtraMappingExpression) GetExpression() string {
	return v.Expression
}

// ReturnTypes returns the CEL expression return types.
func (v *ExtraMappingExpression) ReturnTypes() []*celgo.Type {
	// return types is only used for validation. The claims variable that's available
	// to the claim mapping expressions is a map[string]interface{}, so we can't
	// really know what the return type is during compilation. Strict type checking
	// is done during evaluation.
	return []*celgo.Type{celgo.AnyType}
}

var _ ExpressionAccessor = &UserValidationCondition{}

// UserValidationCondition is a CEL expression that validates a User.
type UserValidationCondition struct {
	Expression string
	Message    string
}

// GetExpression returns the CEL expression.
func (v *UserValidationCondition) GetExpression() string {
	return v.Expression
}

// ReturnTypes returns the CEL expression return types.
func (v *UserValidationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}
