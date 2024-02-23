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

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/traits"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

const (
	attributesVarPrefix = "attributes."
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

type compiler struct {
	envset *environment.EnvSet
}

func newCompiler() *compiler {
	return &compiler{envset: mustBuildEnv()}
}

// CompileCELExpression returns a compiled CEL expression. It evaluates to bool.
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
	if ast.OutputType() != expectedReturnType {
		return resultError(fmt.Sprintf("must evaluate to %v", expectedReturnType.String()), apiservercel.ErrorTypeInvalid)
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
	get func(attr resourceapi.NamedResourcesAttribute) any
}{
	"quantity": {apiservercel.QuantityType, func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.QuantityValue == nil {
			return nil
		}
		return apiservercel.Quantity{Quantity: attr.QuantityValue}
	}},
	"bool": {cel.BoolType, func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.BoolValue == nil {
			return nil
		}
		return *attr.BoolValue
	}},
	"int": {cel.IntType, func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.IntValue == nil {
			return nil
		}
		return *attr.IntValue
	}},
	"intslice": {types.NewListType(cel.IntType), func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.IntSliceValue == nil {
			return nil
		}
		return attr.IntSliceValue.Ints
	}},
	"string": {cel.StringType, func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.StringValue == nil {
			return nil
		}
		return *attr.StringValue
	}},
	"stringslice": {types.NewListType(cel.StringType), func(attr resourceapi.NamedResourcesAttribute) any {
		if attr.StringSliceValue == nil {
			return nil
		}
		return attr.StringSliceValue.Strings
	}},
}

var boolType = reflect.TypeOf(true)

func (c CompilationResult) Evaluate(ctx context.Context, attributes []resourceapi.NamedResourcesAttribute) (bool, error) {
	variables := make(map[string]any, len(valueTypes))
	for name, valueType := range valueTypes {
		variables[attributesVarPrefix+name] = buildValueMapper(c.Environment.CELTypeAdapter(), attributes, valueType.get)
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
	envset := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	versioned := []environment.VersionedOptions{
		{
			IntroducedVersion: version.MajorMinor(1, 30),
			EnvOptions:        buildVersionedAttributes(),
		},
	}
	envset, err := envset.Extend(versioned...)
	if err != nil {
		panic(fmt.Errorf("internal error building CEL environment: %w", err))
	}
	return envset
}

func buildVersionedAttributes() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(valueTypes))
	for name, valueType := range valueTypes {
		options = append(options, cel.Variable(attributesVarPrefix+name, types.NewMapType(cel.StringType, valueType.celType)))
	}
	return options
}

func buildValueMapper(adapter types.Adapter, attributes []resourceapi.NamedResourcesAttribute, get func(resourceapi.NamedResourcesAttribute) any) traits.Mapper {
	// This implementation constructs a map and then let's cel handle the
	// lookup and iteration. This is done for the sake of simplicity.
	// Whether it's faster than writing a custom mapper depends on
	// real-world attribute sets and CEL expressions and would have to be
	// benchmarked.
	valueMap := make(map[string]any)
	for _, attribute := range attributes {
		if value := get(attribute); value != nil {
			valueMap[attribute.Name] = value
		}
	}
	return types.NewStringInterfaceMap(adapter, valueMap)
}
