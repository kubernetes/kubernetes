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

// Package celtest provides a CEL expression evaluator for testing Kubernetes
// admission CEL expressions in Go without a running API server. It supports
// ValidatingAdmissionPolicy, MutatingAdmissionPolicy, and webhook
// matchConditions. The evaluator uses the same CEL environment and compiler
// that the API server uses at runtime, ensuring compilation and evaluation
// parity.
package celtest

import (
	"context"
	"errors"
	"fmt"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	admissioncel "k8s.io/apiserver/pkg/admission/plugin/cel"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
)

// Evaluator compiles and evaluates CEL expressions using the same environment
// the API server configures for admission policies and webhook matchConditions.
type Evaluator struct {
	baseEnvSet        *environment.EnvSet
	version           *version.Version
	preambleVars      []Variable
	costLimit         int64
	patchTypesEnabled bool
	authorizerEnabled bool
}

// Option configures an Evaluator.
type Option func(*Evaluator)

// WithVersion sets the Kubernetes compatibility version for the CEL environment.
// This controls which CEL libraries are available (e.g., Sets at 1.29, IP/CIDR at 1.30).
func WithVersion(major, minor uint) Option {
	return func(e *Evaluator) {
		e.version = version.MajorMinor(major, minor)
	}
}

// WithoutPatchTypes disables mutation-related types (JSONPatch, Object, etc.) in the CEL environment.
func WithoutPatchTypes() Option {
	return func(e *Evaluator) {
		e.patchTypesEnabled = false
	}
}

// WithPreambleVariables adds variables that are compiled and available to all evaluations.
func WithPreambleVariables(vars ...Variable) Option {
	return func(e *Evaluator) {
		e.preambleVars = append(e.preambleVars, vars...)
	}
}

// WithCostLimit overrides the default runtime CEL cost budget.
// Non-positive values are ignored and the default budget is used instead.
func WithCostLimit(limit int64) Option {
	return func(e *Evaluator) {
		e.costLimit = limit
	}
}

// WithoutAuthorizer removes the "authorizer" and "authorizer.requestResource"
// variables from the CEL environment. By default these are declared to match
// the API server's runtime environment.
func WithoutAuthorizer() Option {
	return func(e *Evaluator) {
		e.authorizerEnabled = false
	}
}

// PerCallLimit is the per-expression CEL cost limit used by the API server.
const PerCallLimit = celconfig.PerCallLimit

// Variable is a named CEL expression that can be referenced by other expressions
// as "variables.<name>".
type Variable struct {
	Name       string `yaml:"name"`
	Expression string `yaml:"expression"`
}

// Validation is a single validation rule consisting of a boolean CEL expression,
// a static message, and an optional dynamic message expression.
type Validation struct {
	Path              string
	Expression        string
	Message           string
	MessageExpression string
}

// ValidationSelector identifies a single validation rule for EvalValidation.
// Path selects the rule (e.g., "validations[0]"). Part selects which expression
// to evaluate: "expression" (default) or "messageExpression".
type ValidationSelector struct {
	Path string
	Part string
}

// AdmissionPolicy holds parsed variables, validations, and mutations from a policy YAML.
//
// When constructed manually (not via ParseAdmissionPolicy), the params variable
// is declared by default. Call SetHasParams(false) to disable it, or
// SetHasParams(true) to explicitly enable it, matching the behavior of
// ParseAdmissionPolicy which sets it based on the presence of spec.paramKind.
type AdmissionPolicy struct {
	Variables   []Variable
	Validations []Validation
	Mutations   []Mutation

	hasParams    bool
	hasParamsSet bool
}

// SetHasParams controls whether the "params" variable is declared in the CEL
// environment. When set to true, CEL expressions can reference "params" which
// will resolve to AdmissionInput.Params at evaluation time. When set to false,
// the "params" variable is not declared and referencing it causes a compilation
// error. If SetHasParams is never called on a manually-constructed policy, the
// params variable is declared by default.
func (p *AdmissionPolicy) SetHasParams(has bool) {
	p.hasParams = has
	p.hasParamsSet = true
}

// Mutation is a single mutation operation parsed from a MutatingAdmissionPolicy.
// PatchType is "ApplyConfiguration" or "JSONPatch".
type Mutation struct {
	Path       string
	PatchType  string
	Expression string
}

// MutationSelector identifies a single mutation for EvalMutationByPath.
type MutationSelector struct {
	Path string
}

// MutationResult is the outcome of EvalMutation.
type MutationResult struct {
	Patches []PatchResult
	Cost    int64
}

// PatchResult is the result of evaluating a single mutation expression.
type PatchResult struct {
	Path      string
	PatchType string
	Value     interface{}
	Error     error
}

// AdmissionInput provides the runtime values for CEL evaluation.
// Object and OldObject are unstructured maps; GVK is inferred from their
// apiVersion/kind fields or from Request.Kind if provided.
type AdmissionInput struct {
	Object     map[string]interface{}
	OldObject  map[string]interface{}
	Params     map[string]interface{}
	Request    *admissionv1.AdmissionRequest
	Namespace  *corev1.Namespace
	Authorizer authorizer.Authorizer
}

// AdmissionResult is the outcome of EvalAdmission.
type AdmissionResult struct {
	Allowed    bool
	Violations []Violation
	Cost       int64
}

// Violation describes a single failed validation.
type Violation struct {
	Expression   string
	Message      string
	Error        error
	MessageError error
}

// NewEvaluator creates an Evaluator with the given options. By default it uses
// the current compatibility version with authorizer and patch types enabled.
func NewEvaluator(opts ...Option) (*Evaluator, error) {
	e := &Evaluator{patchTypesEnabled: true, authorizerEnabled: true}
	for _, opt := range opts {
		opt(e)
	}
	if e.version == nil {
		e.version = environment.DefaultCompatibilityVersion()
	}
	e.baseEnvSet = environment.MustBaseEnvSet(e.version)
	return e, nil
}

// CompileCheck validates that a CEL expression compiles against the admission
// environment without evaluating it.
func (e *Evaluator) CompileCheck(expr string) error {
	compiler, decls, err := e.newCompiler(nil)
	if err != nil {
		return err
	}
	evaluator := compiler.CompileMutatingEvaluator(anyExpressionAccessor(expr), decls, e.compileMode())
	return compilationErrors(evaluator.CompilationErrors())
}

// EvalExpression compiles and evaluates a single CEL expression against the
// provided input, returning the raw result value.
func (e *Evaluator) EvalExpression(expr string, input *AdmissionInput) (interface{}, error) {
	compiler, decls, err := e.newCompiler(nil)
	if err != nil {
		return nil, err
	}
	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}
	value, _, err := e.evaluateMutatingWithInputs(compiler, decls, anyExpressionAccessor(expr), evalInputs, e.runtimeCELCostBudget())
	return value, err
}

// EvalAdmission evaluates all validations in the policy against the input and
// returns an AdmissionResult indicating whether the request would be allowed.
func (e *Evaluator) EvalAdmission(policy *AdmissionPolicy, input *AdmissionInput) (*AdmissionResult, error) {
	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}
	if err := e.compileVariableList(compiler, decls, policy.Variables, "variable"); err != nil {
		return nil, err
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}

	accessors := make([]admissioncel.ExpressionAccessor, 0, len(policy.Validations))
	for _, validation := range policy.Validations {
		accessors = append(accessors, boolExpressionAccessor(validation.Expression))
	}
	conditionEvaluator := compiler.CompileCondition(accessors, decls, e.compileMode())
	if err := compilationErrors(conditionEvaluator.CompilationErrors()); err != nil {
		return nil, err
	}

	budget := e.runtimeCELCostBudget()
	results, remaining, err := conditionEvaluator.ForInput(context.Background(), evalInputs.versionedAttr, evalInputs.request, evalInputs.optionalVars, evalInputs.namespace, budget)
	if err != nil {
		return nil, err
	}

	result := &AdmissionResult{Allowed: true}
	for index, validation := range policy.Validations {
		evaluation := results[index]
		if evaluation.Error != nil {
			result.Allowed = false
			result.Violations = append(result.Violations, Violation{Expression: validation.Expression, Error: evaluation.Error})
			continue
		}

		value, ok := evaluationValue(evaluation).(bool)
		if !ok {
			result.Allowed = false
			result.Violations = append(result.Violations, Violation{
				Expression: validation.Expression,
				Error:      fmt.Errorf("validation must return bool, got %T", evaluationValue(evaluation)),
			})
			continue
		}
		if value {
			continue
		}

		result.Allowed = false
		message := validation.Message
		var messageErr error
		if validation.MessageExpression != "" {
			messageValue, nextRemaining, err := e.evaluateMutatingWithInputs(compiler, decls, stringExpressionAccessor(validation.MessageExpression), evalInputs, remaining)
			if nextRemaining >= 0 {
				remaining = nextRemaining
			}
			if err != nil {
				messageErr = err
			} else if resolved, ok := messageValue.(string); ok {
				message = resolved
			}
		}
		result.Violations = append(result.Violations, Violation{Expression: validation.Expression, Message: message, MessageError: messageErr})
	}

	result.Cost = budgetCost(budget, remaining)
	return result, nil
}

// EvalMutation evaluates all mutations in the policy against the input and
// returns a MutationResult containing the raw CEL result for each mutation expression.
// Compilation errors cause an immediate error return (symmetric with EvalAdmission).
// Runtime evaluation errors are recorded per-patch in PatchResult.Error.
func (e *Evaluator) EvalMutation(policy *AdmissionPolicy, input *AdmissionInput) (*MutationResult, error) {
	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}
	if err := e.compileVariableList(compiler, decls, policy.Variables, "variable"); err != nil {
		return nil, err
	}

	// Pre-compile all mutations and fail fast on compilation errors.
	compiled := make([]admissioncel.MutatingEvaluator, 0, len(policy.Mutations))
	for _, mutation := range policy.Mutations {
		evaluator := compiler.CompileMutatingEvaluator(anyExpressionAccessor(mutation.Expression), decls, e.compileMode())
		if err := compilationErrors(evaluator.CompilationErrors()); err != nil {
			return nil, fmt.Errorf("mutation %q: %w", mutation.Path, err)
		}
		compiled = append(compiled, evaluator)
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}

	budget := e.runtimeCELCostBudget()
	result := &MutationResult{}
	for i, mutation := range policy.Mutations {
		ctx := compiler.CreateContext(context.Background())
		evalResult, remaining, err := compiled[i].ForInput(ctx, evalInputs.versionedAttr, evalInputs.request, evalInputs.optionalVars, evalInputs.namespace, budget)
		budget = remaining

		patch := PatchResult{
			Path:      mutation.Path,
			PatchType: mutation.PatchType,
		}
		if err != nil {
			patch.Error = err
		} else if evalResult.Error != nil {
			patch.Error = evalResult.Error
		} else {
			patch.Value = evaluationValue(evalResult)
		}
		result.Patches = append(result.Patches, patch)
	}

	result.Cost = budgetCost(e.runtimeCELCostBudget(), budget)
	return result, nil
}

// EvalMutationByPath evaluates a single mutation selected by path, returning the
// raw CEL result value.
func (e *Evaluator) EvalMutationByPath(policy *AdmissionPolicy, selector MutationSelector, input *AdmissionInput) (interface{}, error) {
	if selector.Path == "" {
		return nil, fmt.Errorf("mutation selector path is required")
	}

	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}
	if err := e.compileVariableList(compiler, decls, policy.Variables, "variable"); err != nil {
		return nil, err
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}

	for _, mutation := range policy.Mutations {
		if mutation.Path != selector.Path {
			continue
		}
		return e.evaluateMutating(compiler, decls, anyExpressionAccessor(mutation.Expression), evalInputs)
	}

	return nil, fmt.Errorf("mutation %q not found in policy", selector.Path)
}

// EvalVariable compiles policy variables up to and including the named variable,
// then evaluates and returns its value.
func (e *Evaluator) EvalVariable(policy *AdmissionPolicy, variableName string, input *AdmissionInput) (interface{}, error) {
	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}

	found := false
	for _, variable := range policy.Variables {
		result := compiler.CompileAndStoreVariable(namedAnyExpressionAccessor(variable.Name, variable.Expression), decls, e.compileMode())
		if result.Error != nil {
			return nil, fmt.Errorf("variable %q: %w", variable.Name, result.Error)
		}
		if variable.Name == variableName {
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("variable %q not found in policy", variableName)
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}
	value, _, err := e.evaluateMutatingWithInputs(compiler, decls, anyExpressionAccessor("variables."+variableName), evalInputs, e.runtimeCELCostBudget())
	return value, err
}

// EvalValidation evaluates a single validation rule selected by path and part,
// returning the raw result value.
func (e *Evaluator) EvalValidation(policy *AdmissionPolicy, selector ValidationSelector, input *AdmissionInput) (interface{}, error) {
	if selector.Path == "" {
		return nil, fmt.Errorf("validation selector path is required")
	}

	part := selector.Part
	if part == "" {
		part = "expression"
	}

	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}
	if err := e.compileVariableList(compiler, decls, policy.Variables, "variable"); err != nil {
		return nil, err
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}

	for _, validation := range policy.Validations {
		if validation.Path != selector.Path {
			continue
		}
		switch part {
		case "expression":
			return e.evaluateMutating(compiler, decls, boolExpressionAccessor(validation.Expression), evalInputs)
		case "messageExpression":
			if validation.MessageExpression == "" {
				return validation.Message, nil
			}
			return e.evaluateMutating(compiler, decls, stringExpressionAccessor(validation.MessageExpression), evalInputs)
		default:
			return nil, fmt.Errorf("unsupported validation part %q", part)
		}
	}

	return nil, fmt.Errorf("validation %q not found in policy", selector.Path)
}

func (e *Evaluator) newCompiler(policy *AdmissionPolicy) (*admissioncel.CompositedCompiler, admissioncel.OptionalVariableDeclarations, error) {
	decls := e.optionalDeclarations(policy)
	compiler, err := admissioncel.NewCompositedCompiler(e.baseEnvSet)
	if err != nil {
		return nil, decls, fmt.Errorf("creating composited compiler: %w", err)
	}
	if err := e.compileVariableList(compiler, decls, e.preambleVars, "preamble variable"); err != nil {
		return nil, decls, err
	}
	return compiler, decls, nil
}

func (e *Evaluator) compileVariableList(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, variables []Variable, label string) error {
	for _, variable := range variables {
		result := compiler.CompileAndStoreVariable(namedAnyExpressionAccessor(variable.Name, variable.Expression), decls, e.compileMode())
		if result.Error != nil {
			return fmt.Errorf("%s %q: %w", label, variable.Name, result.Error)
		}
	}
	return nil
}

func (e *Evaluator) evaluateMutating(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, accessor expressionAccessor, evalInputs *evaluationInputs) (interface{}, error) {
	value, _, err := e.evaluateMutatingWithInputs(compiler, decls, accessor, evalInputs, e.runtimeCELCostBudget())
	return value, err
}

func (e *Evaluator) evaluateMutatingWithInputs(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, accessor expressionAccessor, evalInputs *evaluationInputs, budget int64) (interface{}, int64, error) {
	evaluator := compiler.CompileMutatingEvaluator(accessor, decls, e.compileMode())
	if err := compilationErrors(evaluator.CompilationErrors()); err != nil {
		return nil, budget, err
	}
	ctx := compiler.CreateContext(context.Background())
	result, remaining, err := evaluator.ForInput(ctx, evalInputs.versionedAttr, evalInputs.request, evalInputs.optionalVars, evalInputs.namespace, budget)
	if err != nil {
		return nil, remaining, err
	}
	if result.Error != nil {
		return nil, remaining, result.Error
	}
	return evaluationValue(result), remaining, nil
}

func (e *Evaluator) optionalDeclarations(policy *AdmissionPolicy) admissioncel.OptionalVariableDeclarations {
	hasParams := true
	if policy != nil && policy.hasParamsSet {
		hasParams = policy.hasParams
	}
	return admissioncel.OptionalVariableDeclarations{
		HasParams:     hasParams,
		HasAuthorizer: e.authorizerEnabled,
		HasPatchTypes: e.patchTypesEnabled,
	}
}

func (e *Evaluator) compileMode() environment.Type {
	return environment.NewExpressions
}

func (e *Evaluator) runtimeCELCostBudget() int64 {
	if e.costLimit > 0 {
		return e.costLimit
	}
	return celconfig.RuntimeCELCostBudget
}

func compilationErrors(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	if len(errs) == 1 {
		return errs[0]
	}
	return errors.Join(errs...)
}

func evaluationValue(result admissioncel.EvaluationResult) interface{} {
	if result.EvalResult == nil {
		return nil
	}
	return result.EvalResult.Value()
}

func budgetCost(start, remaining int64) int64 {
	return start - remaining
}

// ParseAdmissionPolicy parses a YAML string into an AdmissionPolicy.
// Supported formats: ValidatingAdmissionPolicy, MutatingAdmissionPolicy,
// webhook configurations (ValidatingWebhookConfiguration,
// MutatingWebhookConfiguration), and flat variables/validations YAML.
func ParseAdmissionPolicy(yamlContent string) (*AdmissionPolicy, error) {
	return parseAdmissionPolicy([]byte(yamlContent))
}

// ParseAdmissionPolicyFile reads and parses a YAML file into an AdmissionPolicy.
// The path is cleaned but not otherwise restricted. Callers are responsible for
// ensuring the path points to a trusted file — do not pass unsanitised user input.
func ParseAdmissionPolicyFile(path string) (*AdmissionPolicy, error) {
	return parseAdmissionPolicyFile(path)
}

// ParseAdmissionInput parses a YAML string into an AdmissionInput. The YAML may
// include object, oldObject, params, request, namespace, and namespaceObject
// fields. Params are passed to CEL as the top-level params object exactly as
// provided.
func ParseAdmissionInput(yamlContent string) (*AdmissionInput, error) {
	return parseAdmissionInput([]byte(yamlContent))
}

// ParseAdmissionInputFile reads and parses a YAML file into an AdmissionInput.
// The path is cleaned but not otherwise restricted. Callers are responsible for
// ensuring the path points to a trusted file - do not pass unsanitised user input.
func ParseAdmissionInputFile(path string) (*AdmissionInput, error) {
	return parseAdmissionInputFile(path)
}

// FormatViolations returns a comma-separated summary of all violation messages and errors.
func (r *AdmissionResult) FormatViolations() string {
	if len(r.Violations) == 0 {
		return ""
	}
	msgs := make([]string, 0, len(r.Violations))
	for _, violation := range r.Violations {
		if violation.Error != nil {
			msgs = append(msgs, fmt.Sprintf("error evaluating %q: %v", violation.Expression, violation.Error))
			continue
		}
		if violation.Message != "" {
			msgs = append(msgs, violation.Message)
		} else {
			msgs = append(msgs, fmt.Sprintf("expression %q evaluated to false", violation.Expression))
		}
	}
	return strings.Join(msgs, ", ")
}
