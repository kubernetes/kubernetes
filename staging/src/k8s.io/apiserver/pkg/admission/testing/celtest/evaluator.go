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

// Package celtest provides a CEL expression evaluator for testing Kubernetes
// admission CEL expressions in Go without a running API server. It supports
// ValidatingAdmissionPolicy, MutatingAdmissionPolicy, and webhook
// matchConditions. The evaluator uses the same CEL environment and compiler
// that the API server uses at runtime, ensuring compilation parity.
//
// EvalValidations and EvalMutation evaluate validations and mutations
// regardless of any MatchConditions on the policy; gating is not enforced
// implicitly. Call EvalMatchConditions separately to assert that a request
// would have been admitted by the policy's match conditions before its rules
// ran.
package celtest

import (
	"context"
	"errors"
	"fmt"
	"strings"

	celtypes "github.com/google/cel-go/common/types"
	"google.golang.org/protobuf/types/known/structpb"
	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	admissioncel "k8s.io/apiserver/pkg/admission/plugin/cel"
	mutatingpatch "k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
)

// Evaluator compiles and evaluates CEL expressions using the same environment
// the API server configures for admission policies and webhook matchConditions.
type Evaluator struct {
	baseEnvSet        *environment.EnvSet
	version           *version.Version
	preambleVars      []variable
	costLimit         int64
	patchTypesEnabled bool
	authorizerEnabled bool
}

// Option configures an Evaluator.
type Option func(*Evaluator)

// PreambleVariable configures a named CEL expression compiled before policy variables.
type PreambleVariable struct {
	Name       string
	Expression string
}

// WithVersion sets the Kubernetes compatibility version for the CEL environment.
// This controls which CEL libraries are available (e.g., Sets at 1.29, IP/CIDR at 1.30).
func WithVersion(major, minor uint) Option {
	return func(e *Evaluator) {
		e.version = version.MajorMinor(major, minor)
	}
}

// WithoutPatchTypes disables mutation-related types (JSONPatch, Object, etc.) for mutation expressions.
func WithoutPatchTypes() Option {
	return func(e *Evaluator) {
		e.patchTypesEnabled = false
	}
}

// WithPreambleVariables adds variables that are compiled and available to all evaluations.
func WithPreambleVariables(vars ...PreambleVariable) Option {
	return func(e *Evaluator) {
		for _, v := range vars {
			e.preambleVars = append(e.preambleVars, variable{Name: v.Name, Expression: v.Expression})
		}
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

// variable is a named CEL expression that can be referenced by other expressions
// as "variables.<name>".
type variable struct {
	Name       string
	Expression string
}

// validation is a single validation rule consisting of a boolean CEL expression,
// a static message, and an optional dynamic message expression.
type validation struct {
	Path              string
	Expression        string
	Message           string
	MessageExpression string
}

// matchCondition is a CEL expression that decides whether a policy or webhook
// should run for a request.
type matchCondition struct {
	Path       string
	Name       string
	Expression string
}

// auditAnnotation is a CEL expression that produces a validating admission
// policy audit annotation value.
type auditAnnotation struct {
	Path            string
	Key             string
	ValueExpression string
}

// AdmissionPolicy holds parsed variables, validations, and mutations from a policy YAML.
// Use ParseAdmissionPolicy or NewFrom* constructors to build an AdmissionPolicy.
type AdmissionPolicy struct {
	variables        []variable
	validations      []validation
	mutations        []mutation
	matchConditions  []matchCondition
	auditAnnotations []auditAnnotation

	hasParams    bool
	hasParamsSet bool
}

// setHasParams controls whether the "params" variable is declared in the CEL
// environment. When set to true, CEL expressions can reference "params" which
// will resolve to AdmissionInput params at evaluation time. When set to false,
// the "params" variable is not declared and referencing it causes a compilation
// error. If setHasParams is never called on a manually-constructed policy, the
// params variable is declared by default.
func (p *AdmissionPolicy) setHasParams(has bool) {
	p.hasParams = has
	p.hasParamsSet = true
}

// mutation is a single mutation operation parsed from a MutatingAdmissionPolicy.
// PatchType is "ApplyConfiguration" or "JSONPatch".
type mutation struct {
	Path       string
	PatchType  string
	Expression string
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
// Use SetObject, SetOldObject, SetParams, or their unstructured variants to set
// CEL object values. Typed values are converted to unstructured objects
// internally. GVK is inferred from the object, oldObject, or Request.Kind when
// provided.
type AdmissionInput struct {
	object     interface{}
	oldObject  interface{}
	params     interface{}
	Request    *admissionv1.AdmissionRequest
	Namespace  *corev1.Namespace
	Authorizer authorizer.Authorizer
}

// NewAdmissionInput returns an empty AdmissionInput for use with setter methods.
func NewAdmissionInput() *AdmissionInput {
	return &AdmissionInput{}
}

// SetObject sets the CEL object variable from a typed Kubernetes object.
func (i *AdmissionInput) SetObject(object runtime.Object) *AdmissionInput {
	i.object = object
	return i
}

// SetUnstructuredObject sets the CEL object variable from an unstructured map.
func (i *AdmissionInput) SetUnstructuredObject(object map[string]interface{}) *AdmissionInput {
	i.object = object
	return i
}

// SetOldObject sets the CEL oldObject variable from a typed Kubernetes object.
func (i *AdmissionInput) SetOldObject(object runtime.Object) *AdmissionInput {
	i.oldObject = object
	return i
}

// SetUnstructuredOldObject sets the CEL oldObject variable from an unstructured map.
func (i *AdmissionInput) SetUnstructuredOldObject(object map[string]interface{}) *AdmissionInput {
	i.oldObject = object
	return i
}

// SetParams sets the CEL params variable from a typed Kubernetes object.
func (i *AdmissionInput) SetParams(params runtime.Object) *AdmissionInput {
	i.params = params
	return i
}

// SetUnstructuredParams sets the CEL params variable from an unstructured map.
func (i *AdmissionInput) SetUnstructuredParams(params map[string]interface{}) *AdmissionInput {
	i.params = params
	return i
}

// AdmissionResult is the allow/deny outcome of evaluating validation rules.
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

// MatchResult is the outcome of evaluating all matchConditions in a policy.
type MatchResult struct {
	Conditions []MatchConditionResult
	Cost       int64
}

// MatchConditionResult is the outcome of evaluating one matchCondition.
type MatchConditionResult struct {
	Path       string
	Name       string
	Expression string
	Value      interface{}
	Error      error
}

// AuditResult is the outcome of evaluating all audit annotation expressions.
type AuditResult struct {
	Annotations []AuditAnnotationResult
	Cost        int64
}

// AuditAnnotationResult is the outcome of evaluating one audit annotation expression.
type AuditAnnotationResult struct {
	Path            string
	Key             string
	ValueExpression string
	Value           interface{}
	Error           error
}

// NewEvaluator creates an Evaluator with the given options. By default it uses
// the current compatibility version with authorizer enabled. Mutation patch
// types are enabled for mutation expressions.
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

type policyEvaluationState struct {
	compiler *admissioncel.CompositedCompiler
	decls    admissioncel.OptionalVariableDeclarations
	inputs   *evaluationInputs
}

func (e *Evaluator) preparePolicyEvaluation(policy *AdmissionPolicy, input *AdmissionInput) (*policyEvaluationState, error) {
	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}
	if err := e.compileVariableList(compiler, decls, policy.variables, "variable", e.evaluationMode()); err != nil {
		return nil, err
	}

	evalInputs, err := buildEvaluationInputs(input)
	if err != nil {
		return nil, err
	}

	return &policyEvaluationState{compiler: compiler, decls: decls, inputs: evalInputs}, nil
}

// CompileCheck validates that a CEL expression compiles against the admission
// environment without evaluating it.
func (e *Evaluator) CompileCheck(expr string) error {
	compiler, decls, err := e.newCompilerWithMode(nil, e.compileCheckMode())
	if err != nil {
		return err
	}
	evaluator := compiler.CompileMutatingEvaluator(anyExpressionAccessor(expr), decls, e.compileCheckMode())
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

// EvalValidations evaluates all validations in the policy against the input and
// returns an AdmissionResult indicating whether the request would be allowed.
func (e *Evaluator) EvalValidations(policy *AdmissionPolicy, input *AdmissionInput) (*AdmissionResult, error) {
	state, err := e.preparePolicyEvaluation(policy, input)
	if err != nil {
		return nil, err
	}

	result := &AdmissionResult{Allowed: true}
	accessors := make([]admissioncel.ExpressionAccessor, 0, len(policy.validations))
	for _, validation := range policy.validations {
		accessors = append(accessors, boolExpressionAccessor(validation.Expression))
	}
	conditionEvaluator := state.compiler.CompileCondition(accessors, state.decls, e.evaluationMode())
	if err := compilationErrors(conditionEvaluator.CompilationErrors()); err != nil {
		return nil, err
	}

	budget := e.runtimeCELCostBudget()
	results, remaining, err := conditionEvaluator.ForInput(context.Background(), state.inputs.versionedAttr, state.inputs.request, state.inputs.optionalVars, state.inputs.namespace, budget)
	if err != nil {
		return nil, err
	}

	for index, validation := range policy.validations {
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
			messageDecls := messageExpressionDeclarations(state.decls)
			messageOptionalVars := messageExpressionOptionalBindings(state.inputs)
			messageValue, nextRemaining, err := e.evaluateMutatingWithBindings(state.compiler, messageDecls, stringExpressionAccessor(validation.MessageExpression), state.inputs, messageOptionalVars, remaining)
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

	result.Cost += budgetCost(budget, remaining)
	return result, nil
}

// EvalMutation evaluates all mutations in the policy against the input and
// returns a MutationResult containing the raw CEL result for each mutation expression.
// Compilation errors cause an immediate error return (symmetric with EvalValidations).
// Runtime evaluation errors are recorded per-patch in PatchResult.Error.
func (e *Evaluator) EvalMutation(policy *AdmissionPolicy, input *AdmissionInput) (*MutationResult, error) {
	state, err := e.preparePolicyEvaluation(policy, input)
	if err != nil {
		return nil, err
	}
	mutationDecls := e.mutationDeclarations(policy)

	// Pre-compile all mutations and fail fast on compilation errors.
	compiled := make([]admissioncel.MutatingEvaluator, 0, len(policy.mutations))
	for _, mutation := range policy.mutations {
		accessor, err := mutationExpressionAccessor(mutation)
		if err != nil {
			return nil, err
		}
		evaluator := state.compiler.CompileMutatingEvaluator(accessor, mutationDecls, e.evaluationMode())
		if err := compilationErrors(evaluator.CompilationErrors()); err != nil {
			return nil, fmt.Errorf("mutation %q: %w", mutation.Path, err)
		}
		compiled = append(compiled, evaluator)
	}

	result := &MutationResult{}
	for i, mutation := range policy.mutations {
		callContext := state.compiler.CreateContext(context.Background())
		budget := e.runtimeCELCostBudget()
		evalResult, remaining, err := compiled[i].ForInput(callContext, state.inputs.versionedAttr, state.inputs.request, state.inputs.optionalVars, state.inputs.namespace, budget)
		result.Cost += budgetCost(budget, remaining)

		patch := PatchResult{
			Path:      mutation.Path,
			PatchType: mutation.PatchType,
		}
		if err != nil {
			patch.Error = err
			result.Patches = append(result.Patches, patch)
			continue
		} else if evalResult.Error != nil {
			patch.Error = evalResult.Error
		} else {
			patch.Value = evaluationValue(evalResult)
		}
		result.Patches = append(result.Patches, patch)
	}

	return result, nil
}

// EvalVariable compiles policy variables up to and including the named variable,
// then evaluates and returns its value.
func (e *Evaluator) EvalVariable(policy *AdmissionPolicy, variableName string, input *AdmissionInput) (interface{}, error) {
	compiler, decls, err := e.newCompiler(policy)
	if err != nil {
		return nil, err
	}

	found := false
	for _, variable := range policy.variables {
		result := compiler.CompileAndStoreVariable(namedAnyExpressionAccessor(variable.Name, variable.Expression), decls, e.evaluationMode())
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

// EvalMatchConditions evaluates all matchConditions in the policy against the input.
func (e *Evaluator) EvalMatchConditions(policy *AdmissionPolicy, input *AdmissionInput) (*MatchResult, error) {
	state, err := e.preparePolicyEvaluation(policy, input)
	if err != nil {
		return nil, err
	}
	return e.evalMatchConditionsWithInputs(state.compiler, state.decls, policy, state.inputs)
}

// EvalAuditAnnotations evaluates all audit annotation expressions in the policy.
func (e *Evaluator) EvalAuditAnnotations(policy *AdmissionPolicy, input *AdmissionInput) (*AuditResult, error) {
	state, err := e.preparePolicyEvaluation(policy, input)
	if err != nil {
		return nil, err
	}
	return e.evalAuditAnnotationsWithInputs(state.compiler, state.decls, policy, state.inputs)
}

func (e *Evaluator) evalMatchConditionsWithInputs(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, policy *AdmissionPolicy, evalInputs *evaluationInputs) (*MatchResult, error) {
	result := &MatchResult{}
	if policy == nil || len(policy.matchConditions) == 0 {
		return result, nil
	}

	accessors := make([]admissioncel.ExpressionAccessor, 0, len(policy.matchConditions))
	for _, condition := range policy.matchConditions {
		accessors = append(accessors, boolExpressionAccessor(condition.Expression))
	}
	conditionEvaluator := compiler.CompileCondition(accessors, decls, e.evaluationMode())
	if err := compilationErrors(conditionEvaluator.CompilationErrors()); err != nil {
		return nil, err
	}

	budget := e.runtimeMatchCELCostBudget()
	evaluations, remaining, err := conditionEvaluator.ForInput(context.Background(), evalInputs.versionedAttr, evalInputs.request, evalInputs.optionalVars, nil, budget)
	result.Cost = budgetCost(budget, remaining)
	if err != nil {
		return nil, err
	}

	for index, condition := range policy.matchConditions {
		evaluation := evaluations[index]
		conditionResult := MatchConditionResult{Path: condition.Path, Name: condition.Name, Expression: condition.Expression}
		if evaluation.Error != nil {
			conditionResult.Error = evaluation.Error
			result.Conditions = append(result.Conditions, conditionResult)
			continue
		}

		value, ok := evaluationValue(evaluation).(bool)
		if !ok {
			conditionResult.Error = fmt.Errorf("matchCondition must return bool, got %T", evaluationValue(evaluation))
			result.Conditions = append(result.Conditions, conditionResult)
			continue
		}
		conditionResult.Value = value
		result.Conditions = append(result.Conditions, conditionResult)
	}
	return result, nil
}

func (e *Evaluator) evalAuditAnnotationsWithInputs(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, policy *AdmissionPolicy, evalInputs *evaluationInputs) (*AuditResult, error) {
	result := &AuditResult{}
	if policy == nil || len(policy.auditAnnotations) == 0 {
		return result, nil
	}

	accessors := make([]admissioncel.ExpressionAccessor, 0, len(policy.auditAnnotations))
	for _, annotation := range policy.auditAnnotations {
		accessors = append(accessors, stringOrNullExpressionAccessor(annotation.ValueExpression))
	}
	conditionEvaluator := compiler.CompileCondition(accessors, decls, e.evaluationMode())
	if err := compilationErrors(conditionEvaluator.CompilationErrors()); err != nil {
		return nil, err
	}

	budget := e.runtimeCELCostBudget()
	optionalVars := auditAnnotationOptionalBindings(evalInputs)
	evaluations, remaining, err := conditionEvaluator.ForInput(context.Background(), evalInputs.versionedAttr, evalInputs.request, optionalVars, evalInputs.namespace, budget)
	result.Cost = budgetCost(budget, remaining)
	if err != nil {
		return nil, err
	}

	for index, annotation := range policy.auditAnnotations {
		evaluation := evaluations[index]
		annotationResult := AuditAnnotationResult{Path: annotation.Path, Key: annotation.Key, ValueExpression: annotation.ValueExpression}
		if evaluation.Error != nil {
			annotationResult.Error = evaluation.Error
			result.Annotations = append(result.Annotations, annotationResult)
			continue
		}

		value := evaluationValue(evaluation)
		switch typed := value.(type) {
		case string:
			annotationResult.Value = strings.TrimSpace(typed)
		case nil:
			annotationResult.Value = nil
		case celtypes.Null:
			annotationResult.Value = nil
		case structpb.NullValue:
			annotationResult.Value = nil
		default:
			annotationResult.Error = fmt.Errorf("valueExpression %q resulted in unsupported return type: %T", annotation.ValueExpression, value)
		}
		result.Annotations = append(result.Annotations, annotationResult)
	}
	return result, nil
}

func (e *Evaluator) newCompiler(policy *AdmissionPolicy) (*admissioncel.CompositedCompiler, admissioncel.OptionalVariableDeclarations, error) {
	return e.newCompilerWithMode(policy, e.evaluationMode())
}

func (e *Evaluator) newCompilerWithMode(policy *AdmissionPolicy, envType environment.Type) (*admissioncel.CompositedCompiler, admissioncel.OptionalVariableDeclarations, error) {
	decls := e.optionalDeclarations(policy)
	compiler, err := admissioncel.NewCompositedCompiler(e.baseEnvSet)
	if err != nil {
		return nil, decls, fmt.Errorf("creating composited compiler: %w", err)
	}
	if err := e.compileVariableList(compiler, decls, e.preambleVars, "preamble variable", envType); err != nil {
		return nil, decls, err
	}
	return compiler, decls, nil
}

func (e *Evaluator) compileVariableList(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, variables []variable, label string, envType environment.Type) error {
	for _, variable := range variables {
		result := compiler.CompileAndStoreVariable(namedAnyExpressionAccessor(variable.Name, variable.Expression), decls, envType)
		if result.Error != nil {
			return fmt.Errorf("%s %q: %w", label, variable.Name, result.Error)
		}
	}
	return nil
}

func (e *Evaluator) evaluateMutatingWithInputs(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, accessor admissioncel.ExpressionAccessor, evalInputs *evaluationInputs, budget int64) (interface{}, int64, error) {
	return e.evaluateMutatingWithBindings(compiler, decls, accessor, evalInputs, evalInputs.optionalVars, budget)
}

func (e *Evaluator) evaluateMutatingWithBindings(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, accessor admissioncel.ExpressionAccessor, evalInputs *evaluationInputs, optionalVars admissioncel.OptionalVariableBindings, budget int64) (interface{}, int64, error) {
	return e.evaluateMutatingWithBindingsAndNamespace(compiler, decls, accessor, evalInputs, optionalVars, evalInputs.namespace, budget)
}

func (e *Evaluator) evaluateMutatingWithBindingsAndNamespace(compiler *admissioncel.CompositedCompiler, decls admissioncel.OptionalVariableDeclarations, accessor admissioncel.ExpressionAccessor, evalInputs *evaluationInputs, optionalVars admissioncel.OptionalVariableBindings, namespace *corev1.Namespace, budget int64) (interface{}, int64, error) {
	evaluator := compiler.CompileMutatingEvaluator(accessor, decls, e.evaluationMode())
	if err := compilationErrors(evaluator.CompilationErrors()); err != nil {
		return nil, budget, err
	}
	ctx := compiler.CreateContext(context.Background())
	result, remaining, err := evaluator.ForInput(ctx, evalInputs.versionedAttr, evalInputs.request, optionalVars, namespace, budget)
	if err != nil {
		return nil, remaining, err
	}
	if result.Error != nil {
		return nil, remaining, result.Error
	}
	return evaluationValue(result), remaining, nil
}

func mutationExpressionAccessor(mutation mutation) (admissioncel.ExpressionAccessor, error) {
	switch mutation.PatchType {
	case string(admissionregistrationv1.PatchTypeApplyConfiguration):
		return &mutatingpatch.ApplyConfigurationCondition{Expression: mutation.Expression}, nil
	case string(admissionregistrationv1.PatchTypeJSONPatch):
		return &mutatingpatch.JSONPatchCondition{Expression: mutation.Expression}, nil
	default:
		return nil, fmt.Errorf("mutation %q unsupported patchType %q", mutation.Path, mutation.PatchType)
	}
}

func messageExpressionDeclarations(decls admissioncel.OptionalVariableDeclarations) admissioncel.OptionalVariableDeclarations {
	decls.HasAuthorizer = false
	return decls
}

func messageExpressionOptionalBindings(evalInputs *evaluationInputs) admissioncel.OptionalVariableBindings {
	return admissioncel.OptionalVariableBindings{VersionedParams: evalInputs.optionalVars.VersionedParams}
}

// auditAnnotationOptionalBindings returns the runtime binding shape used to
// evaluate audit-annotation valueExpressions: params are bound, authorizer is
// stripped.
func auditAnnotationOptionalBindings(evalInputs *evaluationInputs) admissioncel.OptionalVariableBindings {
	return admissioncel.OptionalVariableBindings{VersionedParams: evalInputs.optionalVars.VersionedParams}
}

func (e *Evaluator) optionalDeclarations(policy *AdmissionPolicy) admissioncel.OptionalVariableDeclarations {
	hasParams := true
	if policy != nil && policy.hasParamsSet {
		hasParams = policy.hasParams
	}
	return admissioncel.OptionalVariableDeclarations{
		HasParams:     hasParams,
		HasAuthorizer: e.authorizerEnabled,
	}
}

func (e *Evaluator) mutationDeclarations(policy *AdmissionPolicy) admissioncel.OptionalVariableDeclarations {
	decls := e.optionalDeclarations(policy)
	decls.HasPatchTypes = e.patchTypesEnabled
	return decls
}

func (e *Evaluator) compileCheckMode() environment.Type {
	return environment.NewExpressions
}

func (e *Evaluator) evaluationMode() environment.Type {
	return environment.StoredExpressions
}

func (e *Evaluator) runtimeCELCostBudget() int64 {
	if e.costLimit > 0 {
		return e.costLimit
	}
	return celconfig.RuntimeCELCostBudget
}

func (e *Evaluator) runtimeMatchCELCostBudget() int64 {
	if e.costLimit > 0 {
		return e.costLimit
	}
	return celconfig.RuntimeCELCostBudgetMatchConditions
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
	if remaining < 0 {
		return start
	}
	if remaining > start {
		return 0
	}
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
// The path is passed unchanged to os.ReadFile and is not sanitized; callers
// are responsible for ensuring it points to a trusted file and must not pass
// unsanitised user input.
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
// The path is passed unchanged to os.ReadFile and is not sanitized; callers
// are responsible for ensuring it points to a trusted file and must not pass
// unsanitised user input.
func ParseAdmissionInputFile(path string) (*AdmissionInput, error) {
	return parseAdmissionInputFile(path)
}

// FormatViolations returns a newline-separated summary of all violation messages and errors.
func (r *AdmissionResult) FormatViolations() string {
	if len(r.Violations) == 0 {
		return ""
	}
	msgs := make([]string, 0, len(r.Violations))
	for _, violation := range r.Violations {
		if violation.Error != nil {
			msgs = append(msgs, fmt.Sprintf("error evaluating %q: %v", violation.Expression, violation.Error))
		} else if violation.Message != "" {
			msgs = append(msgs, violation.Message)
		} else {
			msgs = append(msgs, fmt.Sprintf("expression %q evaluated to false", violation.Expression))
		}
		if violation.MessageError != nil {
			msgs = append(msgs, fmt.Sprintf("error evaluating messageExpression for %q: %v", violation.Expression, violation.MessageError))
		}
	}
	return strings.Join(msgs, "\n")
}
