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

package cel

import (
	"fmt"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types/ref"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apimachinery/pkg/util/version"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

const (
	subjectAccessReviewRequestVarName = "request"
)

// CompilationResult represents a compiled authorization cel expression.
type CompilationResult struct {
	Program            cel.Program
	ExpressionAccessor ExpressionAccessor
}

// EvaluationResult contains the minimal required fields and metadata of a cel evaluation
type EvaluationResult struct {
	EvalResult         ref.Val
	ExpressionAccessor ExpressionAccessor
}

// Compiler is an interface for compiling CEL expressions with the desired environment mode.
type Compiler interface {
	CompileCELExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error)
}

type compiler struct {
	envSet *environment.EnvSet
}

// NewCompiler returns a new Compiler.
func NewCompiler(env *environment.EnvSet) Compiler {
	return &compiler{
		envSet: mustBuildEnv(env),
	}
}

func (c compiler) CompileCELExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error) {
	resultError := func(errorString string, errType apiservercel.ErrorType) (CompilationResult, error) {
		err := &apiservercel.Error{
			Type:   errType,
			Detail: errorString,
		}
		return CompilationResult{
			ExpressionAccessor: expressionAccessor,
		}, err
	}
	env, err := c.envSet.Env(environment.StoredExpressions)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}
	ast, issues := env.Compile(expressionAccessor.GetExpression())
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}
	found := false
	returnTypes := expressionAccessor.ReturnTypes()
	for _, returnType := range returnTypes {
		if ast.OutputType() == returnType {
			found = true
			break
		}
	}
	if !found {
		var reason string
		if len(returnTypes) == 1 {
			reason = fmt.Sprintf("must evaluate to %v but got %v", returnTypes[0].String(), ast.OutputType())
		} else {
			reason = fmt.Sprintf("must evaluate to one of %v", returnTypes)
		}

		return resultError(reason, apiservercel.ErrorTypeInvalid)
	}
	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	return CompilationResult{
		Program:            prog,
		ExpressionAccessor: expressionAccessor,
	}, nil
}

func mustBuildEnv(baseEnv *environment.EnvSet) *environment.EnvSet {
	field := func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField {
		return apiservercel.NewDeclField(name, declType, required, nil, nil)
	}
	fields := func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField {
		result := make(map[string]*apiservercel.DeclField, len(fields))
		for _, f := range fields {
			result[f.Name] = f
		}
		return result
	}
	subjectAccessReviewSpecRequestType := buildRequestType(field, fields)
	extended, err := baseEnv.Extend(
		environment.VersionedOptions{
			// we record this as 1.0 since it was available in the
			// first version that supported this feature
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: []cel.EnvOption{
				cel.Variable(subjectAccessReviewRequestVarName, subjectAccessReviewSpecRequestType.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				subjectAccessReviewSpecRequestType,
			},
		},
	)
	if err != nil {
		panic(fmt.Sprintf("environment misconfigured: %v", err))
	}

	return extended
}

// buildRequestType generates a DeclType for SubjectAccessReviewSpec.
// if attributes are added here, also add to convertObjectToUnstructured.
func buildRequestType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	resourceAttributesType := buildResourceAttributesType(field, fields)
	nonResourceAttributesType := buildNonResourceAttributesType(field, fields)
	return apiservercel.NewObjectType("kubernetes.SubjectAccessReviewSpec", fields(
		field("resourceAttributes", resourceAttributesType, false),
		field("nonResourceAttributes", nonResourceAttributesType, false),
		field("user", apiservercel.StringType, false),
		field("groups", apiservercel.NewListType(apiservercel.StringType, -1), false),
		field("extra", apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewListType(apiservercel.StringType, -1), -1), false),
		field("uid", apiservercel.StringType, false),
	))
}

// buildResourceAttributesType generates a DeclType for ResourceAttributes.
// if attributes are added here, also add to convertObjectToUnstructured.
func buildResourceAttributesType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.ResourceAttributes", fields(
		field("namespace", apiservercel.StringType, false),
		field("verb", apiservercel.StringType, false),
		field("group", apiservercel.StringType, false),
		field("version", apiservercel.StringType, false),
		field("resource", apiservercel.StringType, false),
		field("subresource", apiservercel.StringType, false),
		field("name", apiservercel.StringType, false),
	))
}

// buildNonResourceAttributesType generates a DeclType for NonResourceAttributes.
// if attributes are added here, also add to convertObjectToUnstructured.
func buildNonResourceAttributesType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.NonResourceAttributes", fields(
		field("path", apiservercel.StringType, false),
		field("verb", apiservercel.StringType, false),
	))
}

func convertObjectToUnstructured(obj *authorizationv1.SubjectAccessReviewSpec) map[string]interface{} {
	// Construct version containing every SubjectAccessReview user and string attribute field, even omitempty ones, for evaluation by CEL
	extra := obj.Extra
	if extra == nil {
		extra = map[string]authorizationv1.ExtraValue{}
	}
	ret := map[string]interface{}{
		"user":   obj.User,
		"groups": obj.Groups,
		"uid":    string(obj.UID),
		"extra":  extra,
	}
	if obj.ResourceAttributes != nil {
		ret["resourceAttributes"] = map[string]string{
			"namespace":   obj.ResourceAttributes.Namespace,
			"verb":        obj.ResourceAttributes.Verb,
			"group":       obj.ResourceAttributes.Group,
			"version":     obj.ResourceAttributes.Version,
			"resource":    obj.ResourceAttributes.Resource,
			"subresource": obj.ResourceAttributes.Subresource,
			"name":        obj.ResourceAttributes.Name,
		}
	}
	if obj.NonResourceAttributes != nil {
		ret["nonResourceAttributes"] = map[string]string{
			"verb": obj.NonResourceAttributes.Verb,
			"path": obj.NonResourceAttributes.Path,
		}
	}
	return ret
}
