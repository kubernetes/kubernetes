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
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common/types"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
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
	// If true, the compiled expression contains a reference to the identifier "oldSelf".
	UsesOldSelf bool
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
	// NormalizedRuleFieldPath represents the relative fieldPath specified by user after normalization.
	NormalizedRuleFieldPath string
}

// EnvLoader delegates the decision of which CEL environment to use for each expression.
// Callers should return the appropriate CEL environment based on the guidelines from
// environment.NewExpressions and environment.StoredExpressions.
type EnvLoader interface {
	// RuleEnv returns the appropriate environment from the EnvSet for the given CEL rule.
	RuleEnv(envSet *environment.EnvSet, expression string) *cel.Env
	// MessageExpressionEnv returns the appropriate environment from the EnvSet for the given
	// CEL messageExpressions.
	MessageExpressionEnv(envSet *environment.EnvSet, expression string) *cel.Env
}

// NewExpressionsEnvLoader creates an EnvLoader that always uses the NewExpressions environment type.
func NewExpressionsEnvLoader() EnvLoader {
	return alwaysNewEnvLoader{loadFn: func(envSet *environment.EnvSet) *cel.Env {
		return envSet.NewExpressionsEnv()
	}}
}

// StoredExpressionsEnvLoader creates an EnvLoader that always uses the StoredExpressions environment type.
func StoredExpressionsEnvLoader() EnvLoader {
	return alwaysNewEnvLoader{loadFn: func(envSet *environment.EnvSet) *cel.Env {
		return envSet.StoredExpressionsEnv()
	}}
}

type alwaysNewEnvLoader struct {
	loadFn func(envSet *environment.EnvSet) *cel.Env
}

func (pe alwaysNewEnvLoader) RuleEnv(envSet *environment.EnvSet, _ string) *cel.Env {
	return pe.loadFn(envSet)
}

func (pe alwaysNewEnvLoader) MessageExpressionEnv(envSet *environment.EnvSet, _ string) *cel.Env {
	return pe.loadFn(envSet)
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
// baseEnv is used as the base CEL environment, see common.BaseEnvironment.
func Compile(s *schema.Structural, declType *apiservercel.DeclType, perCallLimit uint64, baseEnvSet *environment.EnvSet, envLoader EnvLoader) ([]CompilationResult, error) {
	t := time.Now()
	defer func() {
		metrics.Metrics.ObserveCompilation(time.Since(t))
	}()

	if len(s.Extensions.XValidations) == 0 {
		return nil, nil
	}
	celRules := s.Extensions.XValidations

	oldSelfEnvSet, optionalOldSelfEnvSet, err := prepareEnvSet(baseEnvSet, declType)
	if err != nil {
		return nil, err
	}
	estimator := newCostEstimator(declType)
	// compResults is the return value which saves a list of compilation results in the same order as x-kubernetes-validations rules.
	compResults := make([]CompilationResult, len(celRules))
	maxCardinality := maxCardinality(declType.MinSerializedSize)
	for i, rule := range celRules {
		ruleEnvSet := oldSelfEnvSet
		if rule.OptionalOldSelf != nil && *rule.OptionalOldSelf {
			ruleEnvSet = optionalOldSelfEnvSet
		}
		compResults[i] = compileRule(s, rule, ruleEnvSet, envLoader, estimator, maxCardinality, perCallLimit)
	}

	return compResults, nil
}

func prepareEnvSet(baseEnvSet *environment.EnvSet, declType *apiservercel.DeclType) (oldSelfEnvSet *environment.EnvSet, optionalOldSelfEnvSet *environment.EnvSet, err error) {
	scopedType := declType.MaybeAssignTypeName(generateUniqueSelfTypeName())

	oldSelfEnvSet, err = baseEnvSet.Extend(
		environment.VersionedOptions{
			// Feature epoch was actually 1.23, but we artificially set it to 1.0 because these
			// options should always be present.
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: []cel.EnvOption{
				cel.Variable(ScopedVarName, scopedType.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				scopedType,
			},
		},
		environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 24),
			EnvOptions: []cel.EnvOption{
				cel.Variable(OldScopedVarName, scopedType.CelType()),
			},
		},
	)
	if err != nil {
		return nil, nil, err
	}

	optionalOldSelfEnvSet, err = baseEnvSet.Extend(
		environment.VersionedOptions{
			// Feature epoch was actually 1.23, but we artificially set it to 1.0 because these
			// options should always be present.
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: []cel.EnvOption{
				cel.Variable(ScopedVarName, scopedType.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				scopedType,
			},
		},
		environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 24),
			EnvOptions: []cel.EnvOption{
				cel.Variable(OldScopedVarName, types.NewOptionalType(scopedType.CelType())),
			},
		},
	)
	if err != nil {
		return nil, nil, err
	}

	return oldSelfEnvSet, optionalOldSelfEnvSet, nil
}

func compileRule(s *schema.Structural, rule apiextensions.ValidationRule, envSet *environment.EnvSet, envLoader EnvLoader, estimator *library.CostEstimator, maxCardinality uint64, perCallLimit uint64) (compilationResult CompilationResult) {
	if len(strings.TrimSpace(rule.Rule)) == 0 {
		// include a compilation result, but leave both program and error nil per documented return semantics of this
		// function
		return
	}
	ruleEnv := envLoader.RuleEnv(envSet, rule.Rule)
	ast, issues := ruleEnv.Compile(rule.Rule)
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
			compilationResult.UsesOldSelf = true
			break
		}
	}

	// TODO: Ideally we could configure the per expression limit at validation time and set it to the remaining overall budget, but we would either need a way to pass in a limit at evaluation time or move program creation to validation time
	prog, err := ruleEnv.Program(ast,
		cel.CostLimit(perCallLimit),
		cel.CostTracking(estimator),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "program instantiation failed: " + err.Error()}
		return
	}
	costEst, err := ruleEnv.EstimateCost(ast, estimator)
	if err != nil {
		compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed: " + err.Error()}
		return
	}
	compilationResult.MaxCost = costEst.Max
	compilationResult.MaxCardinality = maxCardinality
	compilationResult.Program = prog
	if rule.MessageExpression != "" {
		messageEnv := envLoader.MessageExpressionEnv(envSet, rule.MessageExpression)
		ast, issues := messageEnv.Compile(rule.MessageExpression)
		if issues != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "messageExpression compilation failed: " + issues.String()}
			return
		}

		supportsMultipleMessageExpressions := messageEnv.HasLibrary(library.AllowMultipleMessageExpressionName)
		outputType := ast.OutputType()

		if outputType != cel.StringType && (!supportsMultipleMessageExpressions || !outputType.IsEquivalentType(cel.ListType(cel.StringType))) {
			detail := "messageExpression must evaluate to a string"
			if supportsMultipleMessageExpressions {
				detail += " or a list of strings"
			}
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: detail}
			return
		}

		_, err := cel.AstToCheckedExpr(ast)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "unexpected messageExpression compilation error: " + err.Error()}
			return
		}

		msgProg, err := messageEnv.Program(ast,
			cel.CostLimit(perCallLimit),
			cel.CostTracking(estimator),
			cel.InterruptCheckFrequency(celconfig.CheckFrequency),
		)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInvalid, Detail: "messageExpression instantiation failed: " + err.Error()}
			return
		}
		costEst, err := messageEnv.EstimateCost(ast, estimator)
		if err != nil {
			compilationResult.MessageExpressionError = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed for messageExpression: " + err.Error()}
			return
		}
		compilationResult.MessageExpression = msgProg
		compilationResult.MessageExpressionMaxCost = costEst.Max
	}
	if rule.FieldPath != "" {
		validFieldPath, _, err := ValidFieldPath(rule.FieldPath, s)
		if err == nil {
			compilationResult.NormalizedRuleFieldPath = validFieldPath.String()
		}
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
