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

package sharingaffinity

import (
	"fmt"
	"sync"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/ext"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
)

const parametersVar = "parameters"

var affinityEnv = sync.OnceValues(func() (*environment.EnvSet, error) {
	return environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()).Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 31),
		EnvOptions: []cel.EnvOption{
			ext.Bindings(ext.BindingsVersion(0)),
			cel.Variable(parametersVar, cel.DynType),
		},
	})
})

type compiledExpression struct {
	env     *cel.Env
	ast     *cel.Ast
	maxCost uint64
	full    cel.Program
}

func compileExpression(expression string, envType environment.Type) (*compiledExpression, error) {
	envset, err := affinityEnv()
	if err != nil {
		return nil, newCompileError(apiservercel.ErrorTypeInternal, fmt.Errorf("build CEL environment: %w", err))
	}

	env, err := envset.Env(envType)
	if err != nil {
		return nil, newCompileError(apiservercel.ErrorTypeInternal, fmt.Errorf("load CEL environment: %w", err))
	}

	ast, issues := env.Compile(expression)
	if issues != nil && issues.Err() != nil {
		return nil, newCompileError(apiservercel.ErrorTypeInvalid, fmt.Errorf("compile CEL expression: %w", issues.Err()))
	}
	if !ast.OutputType().IsExactType(cel.StringType) && !ast.OutputType().IsExactType(cel.DynType) {
		return nil, newCompileError(apiservercel.ErrorTypeInvalid, fmt.Errorf("CEL expression must evaluate to string, not %s", ast.OutputType()))
	}
	costEstimate, err := env.EstimateCost(ast, &library.CostEstimator{SizeEstimator: costEstimator{}})
	if err != nil {
		return nil, newCompileError(apiservercel.ErrorTypeInternal, fmt.Errorf("estimate CEL cost: %w", err))
	}
	compiled := &compiledExpression{env: env, ast: ast, maxCost: costEstimate.Max}
	compiled.full, err = compiled.program(resourceapi.SharingAffinityCELMaxCost)
	if err != nil {
		return nil, newCompileError(apiservercel.ErrorTypeInternal, fmt.Errorf("instantiate CEL program: %w", err))
	}

	return compiled, nil
}

func (e *compiledExpression) program(limit uint64) (cel.Program, error) {
	return e.env.Program(e.ast,
		cel.CostTracking(&library.CostEstimator{}),
		cel.CostLimit(limit),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
}

type costEstimator struct{}

func (costEstimator) EstimateSize(node checker.AstNode) *checker.SizeEstimate {
	if path := node.Path(); len(path) > 0 && path[0] == parametersVar {
		// No schema is available. A field cannot contain more elements or
		// Unicode code points than the enclosing request contains bytes.
		return &checker.SizeEstimate{Min: 0, Max: uint64(celconfig.MaxRequestSizeBytes)}
	}
	return nil
}

func (costEstimator) EstimateCallCost(string, string, *checker.AstNode, []checker.AstNode) *checker.CallEstimate {
	return nil
}

// A budget belongs to one pool/request extraction, never to a cached program.
type budget struct {
	remaining, spent, lastFailedCost uint64
}

func newBudget() *budget {
	return &budget{remaining: resourceapi.SharingAffinityCELMaxCost}
}

func (b *budget) program(compiled *compiledExpression) (cel.Program, error) {
	if b.remaining == resourceapi.SharingAffinityCELMaxCost {
		return compiled.full, nil
	}
	// CEL's limit is fixed at program construction. Re-plan the checked AST,
	// including at zero: a zero-cost constant remains legal at the boundary.
	return compiled.program(b.remaining)
}

func (b *budget) charge(actual uint64) {
	charged := min(actual, b.remaining)
	b.remaining -= charged
	b.spent += charged
}
