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

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/apiserver/pkg/cel/metrics"
)

const (
	// ScopedVarName is the variable name assigned to the locally scoped data element of a CEL validation
	// expression.
	ScopedVarName = "self"

	// OldScopedVarName is the variable name assigned to the existing value of the locally scoped data element of a
	// CEL validation expression.
	OldScopedVarName = "oldSelf"
)

// CompilationResult represents the cel compilation result for one rule
type CompilationResult struct {
	Program cel.Program
	Error   *apiservercel.Error
	// If true, the compiled expression contains a reference to the identifier "oldSelf", and its corresponding rule
	// is implicitly a transition rule.
	TransitionRule bool
	// Represents the worst-case cost of the compiled expression in terms of CEL's cost units, as used by cel.EstimateCost.
	MaxCost uint64
	// MaxCardinality represents the worse case number of times this validation rule could be invoked if contained under an
	// unbounded map or list in an OpenAPIv3 schema.
	MaxCardinality uint64
	// MessageExpression represents the cel Program that should be evaluated to generate an error message if the rule
	// fails to validate. If no MessageExpression was given, or if this expression failed to compile, this will be nil.
	MessageExpression cel.Program
	// MessageExpressionError represents an error encountered during compilation of MessageExpression. If no error was
	// encountered, this will be nil.
	MessageExpressionError *apiservercel.Error
	// MessageExpressionMaxCost represents the worst-case cost of the compiled MessageExpression in terms of CEL's cost units,
	// as used by cel.EstimateCost.
	MessageExpressionMaxCost uint64
}

var (
	initEnvOnce sync.Once
	initEnv     *cel.Env
	initEnvErr  error
)

// This func is duplicated in k8s.io/apiserver/pkg/admission/plugin/cel/validator.go
// If any changes are made here, consider to make the same changes there as well.
func getBaseEnv() (*cel.Env, error) {
	initEnvOnce.Do(func() {
		var opts []cel.EnvOption
		opts = append(opts, cel.HomogeneousAggregateLiterals())
		// Validate function declarations once during base env initialization,
		// so they don't need to be evaluated each time a CEL rule is compiled.
		// This is a relatively expensive operation.
		opts = append(opts, cel.EagerlyValidateDeclarations(true), cel.DefaultUTCTimeZone(true))
		opts = append(opts, library.ExtensionLibs...)

		initEnv, initEnvErr = cel.NewEnv(opts...)
	})
	return initEnv, initEnvErr
}

// Compile compiles all the XValidations rules (without recursing into the schema) and returns a slice containing a
// CompilationResult for each ValidationRule, or an error. declType is expected to be a CEL DeclType corresponding
// to the structural schema.
// Each CompilationResult may contain:
//   - non-nil Program, nil Error: The program was compiled successfully
//   - nil Program, non-nil Error: Compilation resulted in an error
//   - nil Program, nil Error: The provided rule was empty so compilation was not attempted
//
// perCallLimit was added for testing purpose only. Callers should always use const PerCallLimit from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func Compile(s *schema.Structural, declType *apiservercel.DeclType, perCallLimit uint64) ([]CompilationResult, error) {
	t := time.Now()
	defer func() {
		metrics.Metrics.ObserveCompilation(time.Since(t))
	}()

	if len(s.Extensions.XValidations) == 0 {
		return nil, nil
	}
	celRules := s.Extensions.XValidations

	var propDecls []cel.EnvOption
	var root *apiservercel.DeclType
	var ok bool
	baseEnv, err := getBaseEnv()
	if err != nil {
		return nil, err
	}
	reg := apiservercel.NewRegistry(baseEnv)
	scopedTypeName := generateUniqueSelfTypeName()
	rt, err := apiservercel.NewRuleTypes(scopedTypeName, declType, reg)
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
		if declType == nil {
			return nil, fmt.Errorf("rule declared on schema that does not support validation rules type: '%s' x-kubernetes-preserve-unknown-fields: '%t'", s.Type, s.XPreserveUnknownFields)
		}
		root = declType.MaybeAssignTypeName(scopedTypeName)
	}
	propDecls = append(propDecls, cel.Variable(ScopedVarName, root.CelType()))
	propDecls = append(propDecls, cel.Variable(OldScopedVarName, root.CelType()))
	opts = append(opts, propDecls...)
	env, err := baseEnv.Extend(opts...)
	if err != nil {
		return nil, err
	}
	estimator := newCostEstimator(root)
	// compResults is the return value which saves a list of compilation results in the same order as x-kubernetes-validations rules.
	compResults := make([]CompilationResult, len(celRules))
	maxCardinality := maxCardinality(root.MinSerializedSize)
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
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "compilation failed: " + issues.String()}
		return
	}
	if ast.OutputType() != cel.BoolType {
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "cel expression must evaluate to a bool"}
		return
	}

	checkedExpr, err := cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "unexpected compilation error: " + err.Error()}
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
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "program instantiation failed: " + err.Error()}
		return
	}
	costEst, err := env.EstimateCost(ast, estimator)
	if err != nil {
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed: " + err.Error()}
		return
	}
	compilationResult.MaxCost = costEst.Max
	compilationResult.MaxCardinality = maxCardinality
	compilationResult.Program = prog
	if rule.MessageExpression != "" {
		ast, issues := env.Compile(rule.MessageExpression)
		if issues != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "messageExpression compilation failed: " + issues.String()}
			return
		}
		if ast.OutputType() != cel.StringType {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "messageExpression must evaluate to a string"}
			return
		}

		_, err := cel.AstToCheckedExpr(ast)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "unexpected messageExpression compilation error: " + err.Error()}
			return
		}

		msgProg, err := env.Program(ast,
			cel.EvalOptions(cel.OptOptimize, cel.OptTrackCost),
			cel.CostLimit(perCallLimit),
			cel.CostTracking(estimator),
			cel.OptimizeRegex(library.ExtensionLibRegexOptimizations...),
			cel.InterruptCheckFrequency(celconfig.CheckFrequency),
		)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "messageExpression instantiation failed: " + err.Error()}
			return
		}
		costEst, err := env.EstimateCost(ast, estimator)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed for messageExpression: " + err.Error()}
			return
		}
		compilationResult.MessageExpression = msgProg
		compilationResult.MessageExpressionMaxCost = costEst.Max
	}
	return
}

// generateUniqueSelfTypeName creates a placeholder type name to use in a CEL programs for cases
// where we do not wish to expose a stable type name to CEL validator rule authors. For this to effectively prevent
// developers from depending on the generated name (i.e. using it in CEL programs), it must be changed each time a
// CRD is created or updated.
func generateUniqueSelfTypeName() string {
	return fmt.Sprintf("selfType%d", time.Now().Nanosecond())
}

func newCostEstimator(root *apiservercel.DeclType) *library.CostEstimator {
	return &library.CostEstimator{SizeEstimator: &sizeEstimator{root: root}}
}

type sizeEstimator struct {
	root *apiservercel.DeclType
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

// maxCardinality returns the maximum number of times data conforming to the minimum size given could possibly exist in
// an object serialized to JSON. For cases where a schema is contained under map or array schemas of unbounded
// size, this can be used as an estimate as the worst case number of times data matching the schema could be repeated.
// Note that this only assumes a single comma between data elements, so if the schema is contained under only maps,
// this estimates a higher cardinality that would be possible. DeclType.MinSerializedSize is meant to be passed to
// this function.
func maxCardinality(minSize int64) uint64 {
	sz := minSize + 1 // assume at least one comma between elements
	return uint64(celconfig.MaxRequestSizeBytes / sz)
}
