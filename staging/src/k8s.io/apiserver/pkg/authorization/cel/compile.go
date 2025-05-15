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
	celast "github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apimachinery/pkg/util/version"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (
	subjectAccessReviewRequestVarName = "request"

	fieldSelectorVarName = "fieldSelector"
	labelSelectorVarName = "labelSelector"
)

// CompilationResult represents a compiled authorization cel expression.
type CompilationResult struct {
	Program            cel.Program
	ExpressionAccessor ExpressionAccessor

	// These track if a given expression uses fieldSelector and labelSelector,
	// so construction of data passed to the CEL expression can be optimized if those fields are unused.
	UsesFieldSelector bool
	UsesLabelSelector bool
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

// NewDefaultCompiler returns a new Compiler following the default compatibility version.
// Note: the compiler construction depends on feature gates and the compatibility version to be initialized.
func NewDefaultCompiler() Compiler {
	return NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))
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
	checkedExpr, err := cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	celAST, err := celast.ToAST(checkedExpr)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	var usesFieldSelector, usesLabelSelector bool
	celast.PreOrderVisit(celast.NavigateAST(celAST), celast.NewExprVisitor(func(e celast.Expr) {
		// we already know we use both, no need to inspect more
		if usesFieldSelector && usesLabelSelector {
			return
		}

		var fieldName string
		switch e.Kind() {
		case celast.SelectKind:
			// simple select (.fieldSelector / .labelSelector)
			fieldName = e.AsSelect().FieldName()
		case celast.CallKind:
			// optional select (.?fieldSelector / .?labelSelector)
			if e.AsCall().FunctionName() != operators.OptSelect {
				return
			}
			args := e.AsCall().Args()
			// args[0] is the receiver (what comes before the `.?`), args[1] is the field name being optionally selected (what comes after the `.?`)
			if len(args) != 2 || args[1].Kind() != celast.LiteralKind || args[1].AsLiteral().Type() != cel.StringType {
				return
			}
			fieldName, _ = args[1].AsLiteral().Value().(string)
		}

		switch fieldName {
		case fieldSelectorVarName:
			usesFieldSelector = true
		case labelSelectorVarName:
			usesLabelSelector = true
		}
	}))

	prog, err := env.Program(ast)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	return CompilationResult{
		Program:            prog,
		ExpressionAccessor: expressionAccessor,
		UsesFieldSelector:  usesFieldSelector,
		UsesLabelSelector:  usesLabelSelector,
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
func buildResourceAttributesType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	resourceAttributesFields := []*apiservercel.DeclField{
		field("namespace", apiservercel.StringType, false),
		field("verb", apiservercel.StringType, false),
		field("group", apiservercel.StringType, false),
		field("version", apiservercel.StringType, false),
		field("resource", apiservercel.StringType, false),
		field("subresource", apiservercel.StringType, false),
		field("name", apiservercel.StringType, false),
	}
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		resourceAttributesFields = append(resourceAttributesFields, field("fieldSelector", buildFieldSelectorType(field, fields), false))
		resourceAttributesFields = append(resourceAttributesFields, field("labelSelector", buildLabelSelectorType(field, fields), false))
	}
	return apiservercel.NewObjectType("kubernetes.ResourceAttributes", fields(resourceAttributesFields...))
}

func buildFieldSelectorType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.FieldSelectorAttributes", fields(
		field("rawSelector", apiservercel.StringType, false),
		field("requirements", apiservercel.NewListType(buildSelectorRequirementType(field, fields), -1), false),
	))
}

func buildLabelSelectorType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.LabelSelectorAttributes", fields(
		field("rawSelector", apiservercel.StringType, false),
		field("requirements", apiservercel.NewListType(buildSelectorRequirementType(field, fields), -1), false),
	))
}

func buildSelectorRequirementType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.SelectorRequirement", fields(
		field("key", apiservercel.StringType, false),
		field("operator", apiservercel.StringType, false),
		field("values", apiservercel.NewListType(apiservercel.StringType, -1), false),
	))
}

// buildNonResourceAttributesType generates a DeclType for NonResourceAttributes.
func buildNonResourceAttributesType(field func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField, fields func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField) *apiservercel.DeclType {
	return apiservercel.NewObjectType("kubernetes.NonResourceAttributes", fields(
		field("path", apiservercel.StringType, false),
		field("verb", apiservercel.StringType, false),
	))
}
