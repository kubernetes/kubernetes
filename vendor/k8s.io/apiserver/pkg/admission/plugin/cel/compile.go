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
	"fmt"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"sync"

	"github.com/google/cel-go/cel"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

const (
	ObjectVarName                    = "object"
	OldObjectVarName                 = "oldObject"
	ParamsVarName                    = "params"
	RequestVarName                   = "request"
	AuthorizerVarName                = "authorizer"
	RequestResourceAuthorizerVarName = "authorizer.requestResource"
)

var (
	initEnvsOnce sync.Once
	initEnvs     envs
	initEnvsErr  error
)

func getEnvs() (envs, error) {
	initEnvsOnce.Do(func() {
		requiredVarsEnv, err := buildRequiredVarsEnv()
		if err != nil {
			initEnvsErr = err
			return
		}

		initEnvs, err = buildWithOptionalVarsEnvs(requiredVarsEnv)
		if err != nil {
			initEnvsErr = err
			return
		}
	})
	return initEnvs, initEnvsErr
}

// This is a similar code as in k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/compilation.go
// If any changes are made here, consider to make the same changes there as well.
func buildBaseEnv() (*cel.Env, error) {
	var opts []cel.EnvOption
	opts = append(opts, cel.HomogeneousAggregateLiterals())
	// Validate function declarations once during base env initialization,
	// so they don't need to be evaluated each time a CEL rule is compiled.
	// This is a relatively expensive operation.
	opts = append(opts, cel.EagerlyValidateDeclarations(true), cel.DefaultUTCTimeZone(true))
	opts = append(opts, library.ExtensionLibs...)

	return cel.NewEnv(opts...)
}

func buildRequiredVarsEnv() (*cel.Env, error) {
	baseEnv, err := buildBaseEnv()
	if err != nil {
		return nil, err
	}
	var propDecls []cel.EnvOption
	reg := apiservercel.NewRegistry(baseEnv)

	requestType := BuildRequestType()
	rt, err := apiservercel.NewRuleTypes(requestType.TypeName(), requestType, reg)
	if err != nil {
		return nil, err
	}
	if rt == nil {
		return nil, nil
	}
	opts, err := rt.EnvOptions(baseEnv.TypeProvider())
	if err != nil {
		return nil, err
	}
	propDecls = append(propDecls, cel.Variable(ObjectVarName, cel.DynType))
	propDecls = append(propDecls, cel.Variable(OldObjectVarName, cel.DynType))
	propDecls = append(propDecls, cel.Variable(RequestVarName, requestType.CelType()))

	opts = append(opts, propDecls...)
	env, err := baseEnv.Extend(opts...)
	if err != nil {
		return nil, err
	}
	return env, nil
}

type envs map[OptionalVariableDeclarations]*cel.Env

func buildEnvWithVars(baseVarsEnv *cel.Env, options OptionalVariableDeclarations) (*cel.Env, error) {
	var opts []cel.EnvOption
	if options.HasParams {
		opts = append(opts, cel.Variable(ParamsVarName, cel.DynType))
	}
	if options.HasAuthorizer {
		opts = append(opts, cel.Variable(AuthorizerVarName, library.AuthorizerType))
		opts = append(opts, cel.Variable(RequestResourceAuthorizerVarName, library.ResourceCheckType))
	}
	return baseVarsEnv.Extend(opts...)
}

func buildWithOptionalVarsEnvs(requiredVarsEnv *cel.Env) (envs, error) {
	envs := make(envs, 4) // since the number of variable combinations is small, pre-build a environment for each
	for _, hasParams := range []bool{false, true} {
		for _, hasAuthorizer := range []bool{false, true} {
			opts := OptionalVariableDeclarations{HasParams: hasParams, HasAuthorizer: hasAuthorizer}
			env, err := buildEnvWithVars(requiredVarsEnv, opts)
			if err != nil {
				return nil, err
			}
			envs[opts] = env
		}
	}
	return envs, nil
}

// BuildRequestType generates a DeclType for AdmissionRequest. This may be replaced with a utility that
// converts the native type definition to apiservercel.DeclType once such a utility becomes available.
// The 'uid' field is omitted since it is not needed for in-process admission review.
// The 'object' and 'oldObject' fields are omitted since they are exposed as root level CEL variables.
func BuildRequestType() *apiservercel.DeclType {
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
	gvkType := apiservercel.NewObjectType("kubernetes.GroupVersionKind", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("kind", apiservercel.StringType, true),
	))
	gvrType := apiservercel.NewObjectType("kubernetes.GroupVersionResource", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("resource", apiservercel.StringType, true),
	))
	userInfoType := apiservercel.NewObjectType("kubernetes.UserInfo", fields(
		field("username", apiservercel.StringType, false),
		field("uid", apiservercel.StringType, false),
		field("groups", apiservercel.NewListType(apiservercel.StringType, -1), false),
		field("extra", apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewListType(apiservercel.StringType, -1), -1), false),
	))
	return apiservercel.NewObjectType("kubernetes.AdmissionRequest", fields(
		field("kind", gvkType, true),
		field("resource", gvrType, true),
		field("subResource", apiservercel.StringType, false),
		field("requestKind", gvkType, true),
		field("requestResource", gvrType, true),
		field("requestSubResource", apiservercel.StringType, false),
		field("name", apiservercel.StringType, true),
		field("namespace", apiservercel.StringType, false),
		field("operation", apiservercel.StringType, true),
		field("userInfo", userInfoType, true),
		field("dryRun", apiservercel.BoolType, false),
		field("options", apiservercel.DynType, false),
	))
}

// CompilationResult represents a compiled validations expression.
type CompilationResult struct {
	Program            cel.Program
	Error              *apiservercel.Error
	ExpressionAccessor ExpressionAccessor
}

// CompileCELExpression returns a compiled CEL expression.
// perCallLimit was added for testing purpose only. Callers should always use const PerCallLimit from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func CompileCELExpression(expressionAccessor ExpressionAccessor, optionalVars OptionalVariableDeclarations, perCallLimit uint64) CompilationResult {
	var env *cel.Env
	envs, err := getEnvs()
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInternal,
				Detail: "compiler initialization failed: " + err.Error(),
			},
			ExpressionAccessor: expressionAccessor,
		}
	}
	env, ok := envs[optionalVars]
	if !ok {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("compiler initialization failed: failed to load environment for %v", optionalVars),
			},
			ExpressionAccessor: expressionAccessor,
		}
	}

	ast, issues := env.Compile(expressionAccessor.GetExpression())
	if issues != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "compilation failed: " + issues.String(),
			},
			ExpressionAccessor: expressionAccessor,
		}
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
			reason = fmt.Sprintf("must evaluate to %v", returnTypes[0].String())
		} else {
			reason = fmt.Sprintf("must evaluate to one of %v", returnTypes)
		}

		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: reason,
			},
			ExpressionAccessor: expressionAccessor,
		}
	}

	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInternal,
				Detail: "unexpected compilation error: " + err.Error(),
			},
			ExpressionAccessor: expressionAccessor,
		}
	}
	prog, err := env.Program(ast,
		cel.EvalOptions(cel.OptOptimize, cel.OptTrackCost),
		cel.OptimizeRegex(library.ExtensionLibRegexOptimizations...),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
		cel.CostLimit(perCallLimit),
	)
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "program instantiation failed: " + err.Error(),
			},
			ExpressionAccessor: expressionAccessor,
		}
	}
	return CompilationResult{
		Program:            prog,
		ExpressionAccessor: expressionAccessor,
	}
}
