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
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/blang/semver/v4"
	"github.com/google/cel-go/cel"

	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

const (
	driverNameVar = "device.driverName"
	attributesVar = "device.attributes"
)

var (
	Compiler = newCompiler()
)

// CompilationResult represents a compiled expression.
type CompilationResult struct {
	Program     cel.Program
	Error       *apiservercel.Error
	Expression  string
	OutputType  *cel.Type
	Environment *cel.Env
}

// DeviceAttributes defines the input values for a CEL selector expression.
type DeviceAttributes struct {
	// DriverName gets appended to any attribute which does not already have
	// a fully qualified name. If set, then it is also made available as
	// a string attribute.
	DriverName string
	Attributes []resourceapi.DeviceAttribute
}

type compiler struct {
	envset *environment.EnvSet
}

func newCompiler() *compiler {
	return &compiler{envset: mustBuildEnv()}
}

// CompileCELExpression returns a compiled CEL expression. It evaluates to bool.
//
// TODO: validate AST to detect invalid attribute names.
func (c compiler) CompileCELExpression(expression string, envType environment.Type) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
		}
	}

	env, err := c.envset.Env(envType)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}
	expectedReturnType := cel.BoolType
	if ast.OutputType() != expectedReturnType &&
		ast.OutputType() != cel.AnyType {
		return resultError(fmt.Sprintf("must evaluate to %v or the unknown type, not %v", expectedReturnType.String(), ast.OutputType().String()), apiservercel.ErrorTypeInvalid)
	}
	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast,
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	return CompilationResult{
		Program:     prog,
		Expression:  expression,
		OutputType:  ast.OutputType(),
		Environment: env,
	}
}

var valueTypes = map[string]struct {
	celType *cel.Type
	// get returns nil if the attribute doesn't have the type, otherwise
	// the value of that type.
	get func(attr resourceapi.DeviceAttribute) (any, error)
}{
	"int": {apiservercel.QuantityType, func(attr resourceapi.DeviceAttribute) (any, error) {
		if attr.IntValue == nil {
			return nil, nil
		}
		return *attr.IntValue, nil
	}},
	"bool": {cel.BoolType, func(attr resourceapi.DeviceAttribute) (any, error) {
		if attr.BoolValue == nil {
			return nil, nil
		}
		return *attr.BoolValue, nil
	}},
	"string": {cel.StringType, func(attr resourceapi.DeviceAttribute) (any, error) {
		if attr.StringValue == nil {
			return nil, nil
		}
		return *attr.StringValue, nil
	}},
	"version": {SemverType, func(attr resourceapi.DeviceAttribute) (any, error) {
		if attr.VersionValue == nil {
			return nil, nil
		}
		v, err := semver.Parse(*attr.VersionValue)
		if err != nil {
			return nil, fmt.Errorf("parse semantic version: %v", err)
		}

		return Semver{Version: v}, nil
	}},
}

var boolType = reflect.TypeOf(true)

func (c CompilationResult) DeviceMatches(ctx context.Context, input DeviceAttributes) (bool, error) {
	attributes := make(map[string]any, len(input.Attributes))
	for _, attribute := range input.Attributes {
		for _, valueType := range valueTypes {
			value, err := valueType.get(attribute)
			if err != nil {
				return false, err
			}
			if value == nil {
				continue
			}
			name := qualifyAttributeName(attribute.Name, input.DriverName)
			attributes[name] = value
			break
		}
	}

	variables := map[string]any{
		driverNameVar: input.DriverName,
		attributesVar: attributes,
	}

	result, _, err := c.Program.ContextEval(ctx, variables)
	if err != nil {
		return false, err
	}
	resultAny, err := result.ConvertToNative(boolType)
	if err != nil {
		return false, fmt.Errorf("CEL result of type %s could not be converted to bool: %w", result.Type().TypeName(), err)
	}
	resultBool, ok := resultAny.(bool)
	if !ok {
		return false, fmt.Errorf("CEL native result value should have been a bool, got instead: %T", resultAny)
	}
	return resultBool, nil
}

func mustBuildEnv() *environment.EnvSet {
	envset := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), false /* strictCost */)
	versioned := []environment.VersionedOptions{
		{
			// Feature epoch was actually 1.30, but we artificially set it to 1.0 because these
			// options should always be present.
			//
			// TODO (https://github.com/kubernetes/kubernetes/issues/123687): set this
			// version properly before going to beta.
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: append(buildVersionedAttributes(),
				SemverLib(),
			),
		},
	}
	envset, err := envset.Extend(versioned...)
	if err != nil {
		panic(fmt.Errorf("internal error building CEL environment: %w", err))
	}
	return envset
}

func buildVersionedAttributes() []cel.EnvOption {
	options := []cel.EnvOption{
		cel.Variable(driverNameVar, cel.StringType),
		cel.Variable(attributesVar, cel.MapType(cel.StringType, cel.AnyType)),
	}
	return options
}

func qualifyAttributeName(name, domain string) string {
	if domain == "" || strings.Contains(name, "/") {
		return name
	}
	return domain + "/" + name
}
