/*
Copyright The Kubernetes Authors.

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

package affinity

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/ext"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

const objectVar = "object"

type Features struct{}

type compileOptions struct {
	envType   *environment.Type
	costLimit *uint64
}

type compiledExpression struct {
	program    cel.Program
	outputType *cel.Type
	env        *cel.Env
}

func compileExpression(expression string, opts compileOptions) (*compiledExpression, error) {
	envset := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	envset, err := envset.Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 31),
		EnvOptions: []cel.EnvOption{
			ext.Bindings(ext.BindingsVersion(0)),
			cel.Variable(objectVar, cel.DynType),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("build CEL environment: %w", err)
	}

	env, err := envset.Env(environment.StoredExpressions)
	if opts.envType != nil {
		env, err = envset.Env(*opts.envType)
	}
	if err != nil {
		return nil, fmt.Errorf("load CEL environment: %w", err)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return nil, fmt.Errorf("compile CEL expression: %w", issues.Err())
	}

	costLimit := uint64(resourceapi.CELSelectorExpressionMaxCost)
	if opts.costLimit != nil {
		costLimit = *opts.costLimit
	}
	costEstimate, err := env.EstimateCost(ast, costEstimator{})
	if err != nil {
		return nil, fmt.Errorf("estimate CEL cost: %w", err)
	}
	if costEstimate.Max > costLimit {
		return nil, fmt.Errorf("%w: estimated cost %d exceeds limit %d", errCostLimitExceeded, costEstimate.Max, costLimit)
	}
	program, err := env.Program(ast,
		cel.CostLimit(costLimit),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return nil, fmt.Errorf("instantiate CEL program: %w", err)
	}

	return &compiledExpression{
		program:    program,
		outputType: ast.OutputType(),
		env:        env,
	}, nil
}

func (e *compiledExpression) eval(ctx context.Context, object any) (string, *cel.EvalDetails, error) {
	result, details, err := e.program.ContextEval(ctx, map[string]any{objectVar: object})
	if err != nil {
		if strings.Contains(err.Error(), "operation interrupted") && ctx.Err() != nil {
			return "", details, fmt.Errorf("%w: %w", err, context.Cause(ctx))
		}
		return "", details, err
	}

	native, err := result.ConvertToNative(reflect.TypeOf(""))
	if err != nil {
		return "", details, fmt.Errorf("%w: CEL result of type %s could not be converted to string: %w", errNonStringResult, result.Type().TypeName(), err)
	}
	value, ok := native.(string)
	if !ok {
		return "", details, fmt.Errorf("%w: CEL native result value should have been a string, got instead: %T", errNonStringResult, native)
	}
	return value, details, nil
}

type costEstimator struct{}

func (costEstimator) EstimateSize(checker.AstNode) *checker.SizeEstimate {
	return nil
}

func (costEstimator) EstimateCallCost(string, string, *checker.AstNode, []checker.AstNode) *checker.CallEstimate {
	return nil
}
