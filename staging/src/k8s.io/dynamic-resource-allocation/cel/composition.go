/*
Copyright 2026 The Kubernetes Authors.

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
	"math"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/lazy"
	"k8s.io/utils/ptr"
)

const (
	variablesVar      = "variables"
	variablesTypeName = "kubernetes.dra.variables"
)

// NamedVariable is a named CEL sub-expression used with composition.
type NamedVariable struct {
	Name       string
	Expression string
}

// NamedVariablesFromAPI converts API CELSelectorVariable values to NamedVariable.
func NamedVariablesFromAPI(variables []resourceapi.CELSelectorVariable) []NamedVariable {
	if len(variables) == 0 {
		return nil
	}
	out := make([]NamedVariable, len(variables))
	for i, v := range variables {
		out[i] = NamedVariable{Name: v.Name, Expression: v.Expression}
	}
	return out
}

// CompileCELExpressionWithVariables compiles variables in order and then the
// main expression. Variable results are available as variables.<name>.
// Evaluation of variables is lazy (same as ValidatingAdmissionPolicy).
//
// MaxCost is the sum of estimated costs of all variables and the main expression.
// If any variable fails to compile, that error is returned and compilation stops.
func (c compiler) CompileCELExpressionWithVariables(expression string, variables []NamedVariable, options Options) CompilationResult {
	if len(variables) == 0 {
		return c.CompileCELExpression(expression, options)
	}

	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
			MaxCost:    math.MaxUint64,
		}
	}

	variablesMapType := apiservercel.NewObjectType(variablesTypeName, map[string]*apiservercel.DeclField{})
	extendedEnvSet, err := c.envset.Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 0),
		EnvOptions: []cel.EnvOption{
			cel.Variable(variablesVar, variablesMapType.CelType()),
		},
		DeclTypes: []*apiservercel.DeclType{
			variablesMapType,
		},
	})
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error extending CEL environment for variables: %v", err), apiservercel.ErrorTypeInternal)
	}

	state := &compositionState{
		mapType:           variablesMapType,
		compiledVariables: make(map[string]CompilationResult, len(variables)),
		variableOrder:     make([]string, 0, len(variables)),
	}

	var totalCost uint64
	estimateCost := !options.DisableCostEstimation

	for _, v := range variables {
		varResult := c.compileAgainstEnvSet(extendedEnvSet, v.Expression, options, true /* skipOutputTypeCheck */)
		if varResult.Error != nil {
			return CompilationResult{
				Error: &apiservercel.Error{
					Type:   varResult.Error.Type,
					Detail: fmt.Sprintf("variable %q: %s", v.Name, varResult.Error.Detail),
				},
				Expression: expression,
				MaxCost:    math.MaxUint64,
			}
		}
		if estimateCost {
			if totalCost > math.MaxUint64-varResult.MaxCost {
				totalCost = math.MaxUint64
			} else {
				totalCost += varResult.MaxCost
			}
		}
		state.addField(v.Name, varResult.OutputType)
		state.compiledVariables[v.Name] = varResult
		state.variableOrder = append(state.variableOrder, v.Name)
	}

	main := c.compileAgainstEnvSet(extendedEnvSet, expression, options, false /* skipOutputTypeCheck */)
	if main.Error != nil {
		return main
	}
	if estimateCost {
		if totalCost > math.MaxUint64-main.MaxCost {
			main.MaxCost = math.MaxUint64
		} else {
			main.MaxCost = totalCost + main.MaxCost
		}
	} else {
		main.MaxCost = math.MaxUint64
	}
	main.composition = state
	return main
}

func (c compiler) compileAgainstEnvSet(envSet *environment.EnvSet, expression string, options Options, skipOutputTypeCheck bool) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
			MaxCost:    math.MaxUint64,
		}
	}

	envType := ptr.Deref(options.EnvType, environment.StoredExpressions)
	env, err := envSet.Env(envType)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}

	typesFromEnv, err := c.getTypesFromEnv(env)
	if err != nil {
		return resultError("unexpected error loading CEL environment: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	if !skipOutputTypeCheck {
		if outputTypeErr := c.validateOutputType(typesFromEnv, ast.OutputType(), envType, options.DerivedAttribute); outputTypeErr != nil {
			return resultError(outputTypeErr.Detail, outputTypeErr.Type)
		}
	}

	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast,
		cel.CostLimit(ptr.Deref(options.CostLimit, resourceapi.CELSelectorExpressionMaxCost)),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	compilationResult := CompilationResult{
		Program:      prog,
		Expression:   expression,
		OutputType:   ast.OutputType(),
		Environment:  env,
		emptyMapVal:  env.CELTypeAdapter().NativeToValue(map[string]any{}),
		MaxCost:      math.MaxUint64,
		typesFromEnv: typesFromEnv,
	}

	if !options.DisableCostEstimation {
		estimator := c.newCostEstimator(typesFromEnv)
		costEst, err := env.EstimateCost(ast, estimator)
		if err != nil {
			compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed: " + err.Error()}
			return compilationResult
		}
		compilationResult.MaxCost = costEst.Max
	}

	return compilationResult
}

type compositionState struct {
	mapType           *apiservercel.DeclType
	compiledVariables map[string]CompilationResult
	variableOrder     []string
}

func (s *compositionState) addField(name string, celType *cel.Type) {
	s.mapType.Fields[name] = apiservercel.NewDeclField(name, convertCelTypeToDeclType(celType), true, nil, nil)
}

func (s *compositionState) lazyVariables(ctx context.Context, activation map[string]any) ref.Val {
	lazyMap := lazy.NewMapValue(s.mapType)
	for _, name := range s.variableOrder {
		result := s.compiledVariables[name]
		accessor := &variableAccessor{
			name:       name,
			result:     result,
			activation: activation,
			ctx:        ctx,
		}
		lazyMap.Append(name, accessor.callback)
	}
	return lazyMap
}

type variableAccessor struct {
	name       string
	result     CompilationResult
	activation map[string]any
	ctx        context.Context
}

func (a *variableAccessor) callback(_ *lazy.MapValue) ref.Val {
	if a.result.Error != nil {
		return types.NewErr("composited variable %q fails to compile: %v", a.name, a.result.Error)
	}
	v, _, err := a.result.Program.ContextEval(a.ctx, a.activation)
	if err != nil {
		return types.NewErr("composited variable %q fails to evaluate: %v", a.name, err)
	}
	return v
}

func convertCelTypeToDeclType(celType *cel.Type) *apiservercel.DeclType {
	if celType == nil {
		return apiservercel.DynType
	}
	switch celType {
	case cel.AnyType:
		return apiservercel.AnyType
	case cel.BoolType:
		return apiservercel.BoolType
	case cel.BytesType:
		return apiservercel.BytesType
	case cel.DoubleType:
		return apiservercel.DoubleType
	case cel.DurationType:
		return apiservercel.DurationType
	case cel.IntType:
		return apiservercel.IntType
	case cel.NullType:
		return apiservercel.NullType
	case cel.StringType:
		return apiservercel.StringType
	case cel.TimestampType:
		return apiservercel.TimestampType
	case cel.UintType:
		return apiservercel.UintType
	default:
		if celType.HasTrait(traits.ContainerType) && celType.HasTrait(traits.IndexerType) {
			parameters := celType.Parameters()
			switch len(parameters) {
			case 1:
				elemType := convertCelTypeToDeclType(parameters[0])
				return apiservercel.NewListType(elemType, -1)
			case 2:
				keyType := convertCelTypeToDeclType(parameters[0])
				valueType := convertCelTypeToDeclType(parameters[1])
				return apiservercel.NewMapType(keyType, valueType, -1)
			}
		}
		return apiservercel.DynType
	}
}
