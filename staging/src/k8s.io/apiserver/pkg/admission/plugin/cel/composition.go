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
	"context"
	"math"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	v1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/admission"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/lazy"
)

const VariablesTypeName = "kubernetes.variables"

type CompositedCompiler struct {
	Compiler
	FilterCompiler

	CompositionEnv *CompositionEnv
}

type CompositedFilter struct {
	Filter

	compositionEnv *CompositionEnv
}

func NewCompositedCompiler(envSet *environment.EnvSet) (*CompositedCompiler, error) {
	compositionContext, err := NewCompositionEnv(VariablesTypeName, envSet)
	if err != nil {
		return nil, err
	}
	compiler := NewCompiler(compositionContext.EnvSet)
	filterCompiler := NewFilterCompiler(compositionContext.EnvSet)
	return &CompositedCompiler{
		Compiler:       compiler,
		FilterCompiler: filterCompiler,
		CompositionEnv: compositionContext,
	}, nil
}

func (c *CompositedCompiler) CompileAndStoreVariables(variables []NamedExpressionAccessor, options OptionalVariableDeclarations, mode environment.Type) {
	for _, v := range variables {
		_ = c.CompileAndStoreVariable(v, options, mode)
	}
}

func (c *CompositedCompiler) CompileAndStoreVariable(variable NamedExpressionAccessor, options OptionalVariableDeclarations, mode environment.Type) CompilationResult {
	c.CompositionEnv.AddField(variable.GetName())
	result := c.Compiler.CompileCELExpression(variable, options, mode)
	c.CompositionEnv.CompiledVariables[variable.GetName()] = result
	return result
}

func (c *CompositedCompiler) Compile(expressions []ExpressionAccessor, optionalDecls OptionalVariableDeclarations, envType environment.Type) Filter {
	filter := c.FilterCompiler.Compile(expressions, optionalDecls, envType)
	return &CompositedFilter{
		Filter:         filter,
		compositionEnv: c.CompositionEnv,
	}
}

type CompositionEnv struct {
	*environment.EnvSet

	MapType           *apiservercel.DeclType
	CompiledVariables map[string]CompilationResult
}

func (c *CompositionEnv) AddField(name string) {
	c.MapType.Fields[name] = apiservercel.NewDeclField(name, apiservercel.DynType, true, nil, nil)
}

func NewCompositionEnv(typeName string, baseEnvSet *environment.EnvSet) (*CompositionEnv, error) {
	declType := apiservercel.NewObjectType(typeName, map[string]*apiservercel.DeclField{})
	envSet, err := baseEnvSet.Extend(environment.VersionedOptions{
		// set to 1.0 because composition is one of the fundamental components
		IntroducedVersion: version.MajorMinor(1, 0),
		EnvOptions: []cel.EnvOption{
			cel.Variable("variables", declType.CelType()),
		},
		DeclTypes: []*apiservercel.DeclType{
			declType,
		},
	})
	if err != nil {
		return nil, err
	}
	return &CompositionEnv{
		MapType:           declType,
		EnvSet:            envSet,
		CompiledVariables: map[string]CompilationResult{},
	}, nil
}

func (c *CompositionEnv) CreateContext(parent context.Context) CompositionContext {
	return &compositionContext{
		Context:        parent,
		compositionEnv: c,
	}
}

type CompositionContext interface {
	context.Context
	Variables(activation any) ref.Val
	GetAndResetCost() int64
}

type compositionContext struct {
	context.Context

	compositionEnv  *CompositionEnv
	accumulatedCost int64
}

func (c *compositionContext) Variables(activation any) ref.Val {
	lazyMap := lazy.NewMapValue(c.compositionEnv.MapType)
	for name, result := range c.compositionEnv.CompiledVariables {
		accessor := &variableAccessor{
			name:       name,
			result:     result,
			activation: activation,
			context:    c,
		}
		lazyMap.Append(name, accessor.Callback)
	}
	return lazyMap
}

func (f *CompositedFilter) ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *v1.AdmissionRequest, optionalVars OptionalVariableBindings, namespace *corev1.Namespace, runtimeCELCostBudget int64) ([]EvaluationResult, int64, error) {
	ctx = f.compositionEnv.CreateContext(ctx)
	return f.Filter.ForInput(ctx, versionedAttr, request, optionalVars, namespace, runtimeCELCostBudget)
}

func (c *compositionContext) reportCost(cost int64) {
	c.accumulatedCost += cost
}

func (c *compositionContext) GetAndResetCost() int64 {
	cost := c.accumulatedCost
	c.accumulatedCost = 0
	return cost
}

type variableAccessor struct {
	name       string
	result     CompilationResult
	activation any
	context    *compositionContext
}

func (a *variableAccessor) Callback(_ *lazy.MapValue) ref.Val {
	if a.result.Error != nil {
		return types.NewErr("composited variable %q fails to compile: %v", a.name, a.result.Error)
	}

	v, details, err := a.result.Program.Eval(a.activation)
	if details == nil {
		return types.NewErr("unable to get evaluation details of variable %q", a.name)
	}
	costPtr := details.ActualCost()
	if costPtr == nil {
		return types.NewErr("unable to calculate cost of variable %q", a.name)
	}
	cost := int64(*costPtr)
	if *costPtr > math.MaxInt64 {
		cost = math.MaxInt64
	}
	a.context.reportCost(cost)

	if err != nil {
		return types.NewErr("composited variable %q fails to evaluate: %v", a.name, err)
	}
	return v
}
