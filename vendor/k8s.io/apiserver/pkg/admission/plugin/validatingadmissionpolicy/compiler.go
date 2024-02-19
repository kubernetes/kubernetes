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
	"sync"

	"github.com/google/cel-go/cel"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

const (
	ObjectVarName    = "object"
	OldObjectVarName = "oldObject"
	ParamsVarName    = "params"
	RequestVarName   = "request"

	checkFrequency = 100
)

type envs struct {
	noParams   *cel.Env
	withParams *cel.Env
}

var (
	initEnvsOnce sync.Once
	initEnvs     *envs
	initEnvsErr  error
)

func getEnvs() (*envs, error) {
	initEnvsOnce.Do(func() {
		base, err := buildBaseEnv()
		if err != nil {
			initEnvsErr = err
			return
		}
		noParams, err := buildNoParamsEnv(base)
		if err != nil {
			initEnvsErr = err
			return
		}
		withParams, err := buildWithParamsEnv(noParams)
		if err != nil {
			initEnvsErr = err
			return
		}
		initEnvs = &envs{noParams: noParams, withParams: withParams}
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

func buildNoParamsEnv(baseEnv *cel.Env) (*cel.Env, error) {
	var propDecls []cel.EnvOption
	reg := apiservercel.NewRegistry(baseEnv)

	requestType := buildRequestType()
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

func buildWithParamsEnv(noParams *cel.Env) (*cel.Env, error) {
	return noParams.Extend(cel.Variable(ParamsVarName, cel.DynType))
}

// buildRequestType generates a DeclType for AdmissionRequest. This may be replaced with a utility that
// converts the native type definition to apiservercel.DeclType once such a utility becomes available.
// The 'uid' field is omitted since it is not needed for in-process admission review.
// The 'object' and 'oldObject' fields are omitted since they are exposed as root level CEL variables.
func buildRequestType() *apiservercel.DeclType {
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

// CompilationResult represents a compiled ValidatingAdmissionPolicy validation expression.
type CompilationResult struct {
	Program cel.Program
	Error   *apiservercel.Error
}

// CompileValidatingPolicyExpression returns a compiled vaalidating policy CEL expression.
func CompileValidatingPolicyExpression(validationExpression string, hasParams bool) CompilationResult {
	var env *cel.Env
	envs, err := getEnvs()
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInternal,
				Detail: "compiler initialization failed: " + err.Error(),
			},
		}
	}
	if hasParams {
		env = envs.withParams
	} else {
		env = envs.noParams
	}

	ast, issues := env.Compile(validationExpression)
	if issues != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "compilation failed: " + issues.String(),
			},
		}
	}
	if ast.OutputType() != cel.BoolType {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "cel expression must evaluate to a bool",
			},
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
		}
	}
	prog, err := env.Program(ast,
		cel.EvalOptions(cel.OptOptimize),
		cel.OptimizeRegex(library.ExtensionLibRegexOptimizations...),
		cel.InterruptCheckFrequency(checkFrequency),
	)
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "program instantiation failed: " + err.Error(),
			},
		}
	}
	return CompilationResult{
		Program: prog,
	}
}
