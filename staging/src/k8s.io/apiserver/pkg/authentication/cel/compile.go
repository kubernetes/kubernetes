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

	"k8s.io/apimachinery/pkg/util/version"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

const (
	claimsVarName = "claims"
	userVarName   = "user"
)

// compiler implements the Compiler interface.
type compiler struct {
	// varEnvs is a map of CEL environments, keyed by the name of the CEL variable.
	// The CEL variable is available to the expression.
	// We have 2 environments, one for claims and one for user.
	varEnvs map[string]*environment.EnvSet
}

// NewDefaultCompiler returns a new Compiler following the default compatibility version.
// Note: the compiler construction depends on feature gates and the compatibility version to be initialized.
func NewDefaultCompiler() Compiler {
	return NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))
}

// NewCompiler returns a new Compiler.
func NewCompiler(env *environment.EnvSet) Compiler {
	return &compiler{
		varEnvs: mustBuildEnvs(env),
	}
}

// CompileClaimsExpression compiles the given expressionAccessor into a CEL program that can be evaluated.
// The claims CEL variable is available to the expression.
func (c compiler) CompileClaimsExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error) {
	return c.compile(expressionAccessor, claimsVarName)
}

// CompileUserExpression compiles the given expressionAccessor into a CEL program that can be evaluated.
// The user CEL variable is available to the expression.
func (c compiler) CompileUserExpression(expressionAccessor ExpressionAccessor) (CompilationResult, error) {
	return c.compile(expressionAccessor, userVarName)
}

func (c compiler) compile(expressionAccessor ExpressionAccessor, envVarName string) (CompilationResult, error) {
	resultError := func(errorString string, errType apiservercel.ErrorType) (CompilationResult, error) {
		return CompilationResult{}, &apiservercel.Error{
			Type:   errType,
			Detail: errorString,
		}
	}

	env, err := c.varEnvs[envVarName].Env(environment.StoredExpressions)
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
		if ast.OutputType() == returnType || cel.AnyType == returnType {
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

		return resultError(reason, apiservercel.ErrorTypeInvalid)
	}

	if _, err = cel.AstToCheckedExpr(ast); err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	return CompilationResult{
		Program:            prog,
		AST:                ast,
		ExpressionAccessor: expressionAccessor,
	}, nil
}

func buildUserType() *apiservercel.DeclType {
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

	return apiservercel.NewObjectType("kubernetes.UserInfo", fields(
		field("username", apiservercel.StringType, false),
		field("uid", apiservercel.StringType, false),
		field("groups", apiservercel.NewListType(apiservercel.StringType, -1), false),
		field("extra", apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewListType(apiservercel.StringType, -1), -1), false),
	))
}

func mustBuildEnvs(baseEnv *environment.EnvSet) map[string]*environment.EnvSet {
	buildEnvSet := func(envOpts []cel.EnvOption, declTypes []*apiservercel.DeclType) *environment.EnvSet {
		env, err := baseEnv.Extend(environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions:        envOpts,
			DeclTypes:         declTypes,
		})
		if err != nil {
			panic(fmt.Sprintf("environment misconfigured: %v", err))
		}
		return env
	}

	userType := buildUserType()
	claimsType := apiservercel.NewMapType(apiservercel.StringType, apiservercel.AnyType, -1)

	envs := make(map[string]*environment.EnvSet, 2) // build two environments, one for claims and one for user
	envs[claimsVarName] = buildEnvSet([]cel.EnvOption{cel.Variable(claimsVarName, claimsType.CelType())}, []*apiservercel.DeclType{claimsType})
	envs[userVarName] = buildEnvSet([]cel.EnvOption{cel.Variable(userVarName, userType.CelType())}, []*apiservercel.DeclType{userType})

	return envs
}
