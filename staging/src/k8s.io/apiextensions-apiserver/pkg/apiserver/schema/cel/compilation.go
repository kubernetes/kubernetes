/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"
	"sync"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/checker/decls"
	expr "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	"google.golang.org/protobuf/proto"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/library"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/metrics"
	celmodel "k8s.io/apiextensions-apiserver/third_party/forked/celopenapi/model"
)

const (
	// ScopedVarName is the variable name assigned to the locally scoped data element of a CEL validation
	// expression.
	ScopedVarName = "self"

	// OldScopedVarName is the variable name assigned to the existing value of the locally scoped data element of a
	// CEL validation expression.
	OldScopedVarName = "oldSelf"

	// PerCallLimit specify the actual cost limit per CEL validation call
	// current PerCallLimit gives roughly 0.1 second for each expression validation call
	PerCallLimit = 1000000

	// RuntimeCELCostBudget is the overall cost budget for runtime CEL validation cost per CustomResource
	// current RuntimeCELCostBudget gives roughly 1 seconds for CR validation
	RuntimeCELCostBudget = 10000000

	// checkFrequency configures the number of iterations within a comprehension to evaluate
	// before checking whether the function evaluation has been interrupted
	checkFrequency = 100
)

// CompilationResult represents the cel compilation result for one rule
type CompilationResult struct {
	Program cel.Program
	Error   *Error
	// If true, the compiled expression contains a reference to the identifier "oldSelf", and its corresponding rule
	// is implicitly a transition rule.
	TransitionRule bool
	// Represents the worst-case cost of the compiled expression in terms of CEL's cost units, as used by cel.EstimateCost.
	MaxCost uint64
	// MaxCardinality represents the worse case number of times this validation rule could be invoked if contained under an
	// unbounded map or list in an OpenAPIv3 schema.
	MaxCardinality uint64
}

var (
	initEnvOnce sync.Once
	initEnv     *cel.Env
	initEnvErr  error
)

func getBaseEnv() (*cel.Env, error) {
	initEnvOnce.Do(func() {
		var opts []cel.EnvOption
		opts = append(opts, cel.HomogeneousAggregateLiterals())
		// Validate function declarations once during base env initialization,
		// so they don't need to be evaluated each time a CEL rule is compiled.
		// This is a relatively expensive operation.
		opts = append(opts, cel.EagerlyValidateDeclarations(true))
		opts = append(opts, library.ExtensionLibs...)

		initEnv, initEnvErr = cel.NewEnv(opts...)
	})
	return initEnv, initEnvErr
}

// Compile compiles all the XValidations rules (without recursing into the schema) and returns a slice containing a
// CompilationResult for each ValidationRule, or an error.
// Each CompilationResult may contain:
/// - non-nil Program, nil Error: The program was compiled successfully
//  - nil Program, non-nil Error: Compilation resulted in an error
//  - nil Program, nil Error: The provided rule was empty so compilation was not attempted
// perCallLimit was added for testing purpose only. Callers should always use const PerCallLimit as input.
func Compile(s *schema.Structural, isResourceRoot bool, perCallLimit uint64) ([]CompilationResult, error) {
	t := time.Now()
	defer metrics.Metrics.ObserveCompilation(time.Since(t))

	if len(s.Extensions.XValidations) == 0 {
		return nil, nil
	}
	celRules := s.Extensions.XValidations

	var propDecls []*expr.Decl
	var root *celmodel.DeclType
	var ok bool
	baseEnv, err := getBaseEnv()
	if err != nil {
		return nil, err
	}
	reg := celmodel.NewRegistry(baseEnv)
	scopedTypeName := generateUniqueSelfTypeName()
	rt, err := celmodel.NewRuleTypes(scopedTypeName, s, isResourceRoot, reg)
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
	root, ok = rt.FindDeclType(scopedTypeName)
	if !ok {
		rootDecl := celmodel.SchemaDeclType(s, isResourceRoot)
		if rootDecl == nil {
			return nil, fmt.Errorf("rule declared on schema that does not support validation rules type: '%s' x-kubernetes-preserve-unknown-fields: '%t'", s.Type, s.XPreserveUnknownFields)
		}
		root = rootDecl.MaybeAssignTypeName(scopedTypeName)
	}
	propDecls = append(propDecls, decls.NewVar(ScopedVarName, root.ExprType()))
	propDecls = append(propDecls, decls.NewVar(OldScopedVarName, root.ExprType()))
	opts = append(opts, cel.Declarations(propDecls...))
	env, err := baseEnv.Extend(opts...)
	if err != nil {
		return nil, err
	}
	estimator := newCostEstimator(root)
	// compResults is the return value which saves a list of compilation results in the same order as x-kubernetes-validations rules.
	compResults := make([]CompilationResult, len(celRules))
	maxCardinality := celmodel.MaxCardinality(s)
	for i, rule := range celRules {
		compResults[i] = compileRule(rule, env, perCallLimit, estimator, maxCardinality)
	}

	return compResults, nil
}

func compileRule(rule apiextensions.ValidationRule, env *cel.Env, perCallLimit uint64, estimator *library.CostEstimator, maxCardinality uint64) (compilationResult CompilationResult) {
	if len(strings.TrimSpace(rule.Rule)) == 0 {
		// include a compilation result, but leave both program and error nil per documented return semantics of this
		// function
		return
	}
	ast, issues := env.Compile(rule.Rule)
	if issues != nil {
		compilationResult.Error = &Error{ErrorTypeInvalid, "compilation failed: " + issues.String()}
		return
	}
	if !proto.Equal(ast.ResultType(), decls.Bool) {
		compilationResult.Error = &Error{ErrorTypeInvalid, "cel expression must evaluate to a bool"}
		return
	}

	checkedExpr, err := cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		compilationResult.Error = &Error{ErrorTypeInternal, "unexpected compilation error: " + err.Error()}
		return
	}
	for _, ref := range checkedExpr.ReferenceMap {
		if ref.Name == OldScopedVarName {
			compilationResult.TransitionRule = true
			break
		}
	}

	// TODO: Ideally we could configure the per expression limit at validation time and set it to the remaining overall budget, but we would either need a way to pass in a limit at evaluation time or move program creation to validation time
	prog, err := env.Program(ast,
		cel.EvalOptions(cel.OptOptimize, cel.OptTrackCost),
		cel.CostLimit(perCallLimit),
		cel.CostTracking(estimator),
		cel.OptimizeRegex(library.ExtensionLibRegexOptimizations...),
		cel.InterruptCheckFrequency(checkFrequency),
	)
	if err != nil {
		compilationResult.Error = &Error{ErrorTypeInvalid, "program instantiation failed: " + err.Error()}
		return
	}
	costEst, err := env.EstimateCost(ast, estimator)
	if err != nil {
		compilationResult.Error = &Error{ErrorTypeInternal, "cost estimation failed: " + err.Error()}
		return
	}
	compilationResult.MaxCost = costEst.Max
	compilationResult.MaxCardinality = maxCardinality
	compilationResult.Program = prog
	return
}

// generateUniqueSelfTypeName creates a placeholder type name to use in a CEL programs for cases
// where we do not wish to expose a stable type name to CEL validator rule authors. For this to effectively prevent
// developers from depending on the generated name (i.e. using it in CEL programs), it must be changed each time a
// CRD is created or updated.
func generateUniqueSelfTypeName() string {
	return fmt.Sprintf("selfType%d", time.Now().Nanosecond())
}

func newCostEstimator(root *celmodel.DeclType) *library.CostEstimator {
	return &library.CostEstimator{SizeEstimator: &sizeEstimator{root: root}}
}

type sizeEstimator struct {
	root *celmodel.DeclType
}

func (c *sizeEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
	if len(element.Path()) == 0 {
		// Path() can return an empty list, early exit if it does since we can't
		// provide size estimates when that happens
		return nil
	}
	currentNode := c.root
	// cut off "self" from path, since we always start there
	for _, name := range element.Path()[1:] {
		switch name {
		case "@items", "@values":
			if currentNode.ElemType == nil {
				return nil
			}
			currentNode = currentNode.ElemType
		case "@keys":
			if currentNode.KeyType == nil {
				return nil
			}
			currentNode = currentNode.KeyType
		default:
			field, ok := currentNode.Fields[name]
			if !ok {
				return nil
			}
			if field.Type == nil {
				return nil
			}
			currentNode = field.Type
		}
	}
	return &checker.SizeEstimate{Min: 0, Max: uint64(currentNode.MaxElements)}
}

func (c *sizeEstimator) EstimateCallCost(function, overloadID string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	return nil
}
