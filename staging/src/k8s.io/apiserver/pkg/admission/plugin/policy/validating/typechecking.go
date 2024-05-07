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

package validating

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/cel-go/cel"

	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/klog/v2"
)

const maxTypesToCheck = 10

type TypeChecker struct {
	SchemaResolver resolver.SchemaResolver
	RestMapper     meta.RESTMapper
}

// TypeCheckingContext holds information about the policy being type-checked.
// The struct is opaque to the caller.
type TypeCheckingContext struct {
	gvks          []schema.GroupVersionKind
	declTypes     []*apiservercel.DeclType
	paramGVK      schema.GroupVersionKind
	paramDeclType *apiservercel.DeclType

	variables []v1beta1.Variable
}

type typeOverwrite struct {
	object *apiservercel.DeclType
	params *apiservercel.DeclType
}

// TypeCheckingResult holds the issues found during type checking, any returned
// error, and the gvk that the type checking is performed against.
type TypeCheckingResult struct {
	// GVK is the associated GVK
	GVK schema.GroupVersionKind
	// Issues contain machine-readable information about the typechecking result.
	Issues error
	// Err is the possible error that was encounter during type checking.
	Err error
}

// TypeCheckingResults is a collection of TypeCheckingResult
type TypeCheckingResults []*TypeCheckingResult

func (rs TypeCheckingResults) String() string {
	var messages []string
	for _, r := range rs {
		message := r.String()
		if message != "" {
			messages = append(messages, message)
		}
	}
	return strings.Join(messages, "\n")
}

// String converts the result to human-readable form as a string.
func (r *TypeCheckingResult) String() string {
	if r.Issues == nil && r.Err == nil {
		return ""
	}
	if r.Err != nil {
		return fmt.Sprintf("%v: type checking error: %v\n", r.GVK, r.Err)
	}
	return fmt.Sprintf("%v: %s\n", r.GVK, r.Issues)
}

// Check preforms the type check against the given policy, and format the result
// as []ExpressionWarning that is ready to be set in policy.Status
// The result is nil if type checking returns no warning.
// The policy object is NOT mutated. The caller should update Status accordingly
func (c *TypeChecker) Check(policy *v1beta1.ValidatingAdmissionPolicy) []v1beta1.ExpressionWarning {
	ctx := c.CreateContext(policy)

	// warnings to return, note that the capacity is optimistically set to zero
	var warnings []v1beta1.ExpressionWarning // intentionally not setting capacity

	// check main validation expressions and their message expressions, located in spec.validations[*]
	fieldRef := field.NewPath("spec", "validations")
	for i, v := range policy.Spec.Validations {
		results := c.CheckExpression(ctx, v.Expression)
		if len(results) != 0 {
			warnings = append(warnings, v1beta1.ExpressionWarning{
				FieldRef: fieldRef.Index(i).Child("expression").String(),
				Warning:  results.String(),
			})
		}
		// Note that MessageExpression is optional
		if v.MessageExpression == "" {
			continue
		}
		results = c.CheckExpression(ctx, v.MessageExpression)
		if len(results) != 0 {
			warnings = append(warnings, v1beta1.ExpressionWarning{
				FieldRef: fieldRef.Index(i).Child("messageExpression").String(),
				Warning:  results.String(),
			})
		}
	}

	return warnings
}

// CreateContext resolves all types and their schemas from a policy definition and creates the context.
func (c *TypeChecker) CreateContext(policy *v1beta1.ValidatingAdmissionPolicy) *TypeCheckingContext {
	ctx := new(TypeCheckingContext)
	allGvks := c.typesToCheck(policy)
	gvks := make([]schema.GroupVersionKind, 0, len(allGvks))
	declTypes := make([]*apiservercel.DeclType, 0, len(allGvks))
	for _, gvk := range allGvks {
		declType, err := c.declType(gvk)
		if err != nil {
			// type checking errors MUST NOT alter the behavior of the policy
			// even if an error occurs.
			if !errors.Is(err, resolver.ErrSchemaNotFound) {
				// Anything except ErrSchemaNotFound is an internal error
				klog.V(2).ErrorS(err, "internal error: schema resolution failure", "gvk", gvk)
			}
			// skip for not found or internal error
			continue
		}
		gvks = append(gvks, gvk)
		declTypes = append(declTypes, declType)
	}
	ctx.gvks = gvks
	ctx.declTypes = declTypes

	paramsGVK := c.paramsGVK(policy) // maybe empty, correctly handled
	paramsDeclType, err := c.declType(paramsGVK)
	if err != nil {
		if !errors.Is(err, resolver.ErrSchemaNotFound) {
			klog.V(2).ErrorS(err, "internal error: cannot resolve schema for params", "gvk", paramsGVK)
		}
		paramsDeclType = nil
	}
	ctx.paramGVK = paramsGVK
	ctx.paramDeclType = paramsDeclType
	ctx.variables = policy.Spec.Variables
	return ctx
}

func (c *TypeChecker) compiler(ctx *TypeCheckingContext, typeOverwrite typeOverwrite) (*plugincel.CompositedCompiler, error) {
	envSet, err := buildEnvSet(
		/* hasParams */ ctx.paramDeclType != nil,
		/* hasAuthorizer */ true,
		typeOverwrite)
	if err != nil {
		return nil, err
	}
	env, err := plugincel.NewCompositionEnv(plugincel.VariablesTypeName, envSet)
	if err != nil {
		return nil, err
	}
	compiler := &plugincel.CompositedCompiler{
		Compiler:       &typeCheckingCompiler{typeOverwrite: typeOverwrite, compositionEnv: env},
		CompositionEnv: env,
	}
	return compiler, nil
}

// CheckExpression type checks a single expression, given the context
func (c *TypeChecker) CheckExpression(ctx *TypeCheckingContext, expression string) TypeCheckingResults {
	var results TypeCheckingResults
	for i, gvk := range ctx.gvks {
		declType := ctx.declTypes[i]
		compiler, err := c.compiler(ctx, typeOverwrite{
			object: declType,
			params: ctx.paramDeclType,
		})
		if err != nil {
			utilruntime.HandleError(err)
			continue
		}
		options := plugincel.OptionalVariableDeclarations{
			HasParams:     ctx.paramDeclType != nil,
			HasAuthorizer: true,
		}
		compiler.CompileAndStoreVariables(convertv1beta1Variables(ctx.variables), options, environment.StoredExpressions)
		result := compiler.CompileCELExpression(celExpression(expression), options, environment.StoredExpressions)
		if err := result.Error; err != nil {
			typeCheckingResult := &TypeCheckingResult{GVK: gvk}
			if err.Type == apiservercel.ErrorTypeInvalid {
				typeCheckingResult.Issues = err
			} else {
				typeCheckingResult.Err = err
			}
			results = append(results, typeCheckingResult)
		}
	}
	return results
}

type celExpression string

func (c celExpression) GetExpression() string {
	return string(c)
}

func (c celExpression) ReturnTypes() []*cel.Type {
	return []*cel.Type{cel.AnyType}
}
func generateUniqueTypeName(kind string) string {
	return fmt.Sprintf("%s%d", kind, time.Now().Nanosecond())
}

func (c *TypeChecker) declType(gvk schema.GroupVersionKind) (*apiservercel.DeclType, error) {
	if gvk.Empty() {
		return nil, nil
	}
	s, err := c.SchemaResolver.ResolveSchema(gvk)
	if err != nil {
		return nil, err
	}
	return common.SchemaDeclType(&openapi.Schema{Schema: s}, true).MaybeAssignTypeName(generateUniqueTypeName(gvk.Kind)), nil
}

func (c *TypeChecker) paramsGVK(policy *v1beta1.ValidatingAdmissionPolicy) schema.GroupVersionKind {
	if policy.Spec.ParamKind == nil {
		return schema.GroupVersionKind{}
	}
	gv, err := schema.ParseGroupVersion(policy.Spec.ParamKind.APIVersion)
	if err != nil {
		return schema.GroupVersionKind{}
	}
	return gv.WithKind(policy.Spec.ParamKind.Kind)
}

// typesToCheck extracts a list of GVKs that needs type checking from the policy
// the result is sorted in the order of Group, Version, and Kind
func (c *TypeChecker) typesToCheck(p *v1beta1.ValidatingAdmissionPolicy) []schema.GroupVersionKind {
	gvks := sets.New[schema.GroupVersionKind]()
	if p.Spec.MatchConstraints == nil || len(p.Spec.MatchConstraints.ResourceRules) == 0 {
		return nil
	}
	restMapperRefreshAttempted := false // at most once per policy, refresh RESTMapper and retry resolution.
	for _, rule := range p.Spec.MatchConstraints.ResourceRules {
		groups := extractGroups(&rule.Rule)
		if len(groups) == 0 {
			continue
		}
		versions := extractVersions(&rule.Rule)
		if len(versions) == 0 {
			continue
		}
		resources := extractResources(&rule.Rule)
		if len(resources) == 0 {
			continue
		}
		// sort GVRs so that the loop below provides
		// consistent results.
		sort.Strings(groups)
		sort.Strings(versions)
		sort.Strings(resources)
		count := 0
		for _, group := range groups {
			for _, version := range versions {
				for _, resource := range resources {
					gvr := schema.GroupVersionResource{
						Group:    group,
						Version:  version,
						Resource: resource,
					}
					resolved, err := c.RestMapper.KindsFor(gvr)
					if err != nil {
						if restMapperRefreshAttempted {
							// RESTMapper refresh happens at most once per policy
							continue
						}
						c.tryRefreshRESTMapper()
						restMapperRefreshAttempted = true
						resolved, err = c.RestMapper.KindsFor(gvr)
						if err != nil {
							continue
						}
					}
					for _, r := range resolved {
						if !r.Empty() {
							gvks.Insert(r)
							count++
							// early return if maximum number of types are already
							// collected
							if count == maxTypesToCheck {
								if gvks.Len() == 0 {
									return nil
								}
								return sortGVKList(gvks.UnsortedList())
							}
						}
					}
				}
			}
		}
	}
	if gvks.Len() == 0 {
		return nil
	}
	return sortGVKList(gvks.UnsortedList())
}

func extractGroups(rule *v1beta1.Rule) []string {
	groups := make([]string, 0, len(rule.APIGroups))
	for _, group := range rule.APIGroups {
		// give up if wildcard
		if strings.ContainsAny(group, "*") {
			return nil
		}
		groups = append(groups, group)
	}
	return groups
}

func extractVersions(rule *v1beta1.Rule) []string {
	versions := make([]string, 0, len(rule.APIVersions))
	for _, version := range rule.APIVersions {
		if strings.ContainsAny(version, "*") {
			return nil
		}
		versions = append(versions, version)
	}
	return versions
}

func extractResources(rule *v1beta1.Rule) []string {
	resources := make([]string, 0, len(rule.Resources))
	for _, resource := range rule.Resources {
		// skip wildcard and subresources
		if strings.ContainsAny(resource, "*/") {
			continue
		}
		resources = append(resources, resource)
	}
	return resources
}

// sortGVKList sorts the list by Group, Version, and Kind
// returns the list itself.
func sortGVKList(list []schema.GroupVersionKind) []schema.GroupVersionKind {
	sort.Slice(list, func(i, j int) bool {
		if g := strings.Compare(list[i].Group, list[j].Group); g != 0 {
			return g < 0
		}
		if v := strings.Compare(list[i].Version, list[j].Version); v != 0 {
			return v < 0
		}
		return strings.Compare(list[i].Kind, list[j].Kind) < 0
	})
	return list
}

// tryRefreshRESTMapper refreshes the RESTMapper if it supports refreshing.
func (c *TypeChecker) tryRefreshRESTMapper() {
	if r, ok := c.RestMapper.(meta.ResettableRESTMapper); ok {
		r.Reset()
	}
}

func buildEnvSet(hasParams bool, hasAuthorizer bool, types typeOverwrite) (*environment.EnvSet, error) {
	baseEnv := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	requestType := plugincel.BuildRequestType()
	namespaceType := plugincel.BuildNamespaceType()

	var varOpts []cel.EnvOption
	var declTypes []*apiservercel.DeclType

	// namespace, hand-crafted type
	declTypes = append(declTypes, namespaceType)
	varOpts = append(varOpts, createVariableOpts(namespaceType, plugincel.NamespaceVarName)...)

	// request, hand-crafted type
	declTypes = append(declTypes, requestType)
	varOpts = append(varOpts, createVariableOpts(requestType, plugincel.RequestVarName)...)

	// object and oldObject, same type, type(s) resolved from constraints
	declTypes = append(declTypes, types.object)
	varOpts = append(varOpts, createVariableOpts(types.object, plugincel.ObjectVarName, plugincel.OldObjectVarName)...)

	// params, defined by ParamKind
	if hasParams && types.params != nil {
		declTypes = append(declTypes, types.params)
		varOpts = append(varOpts, createVariableOpts(types.params, plugincel.ParamsVarName)...)
	}

	// authorizer, implicitly available to all expressions of a policy
	if hasAuthorizer {
		// we only need its structure but not the variable itself
		varOpts = append(varOpts, cel.Variable("authorizer", library.AuthorizerType))
	}

	return baseEnv.Extend(
		environment.VersionedOptions{
			// Feature epoch was actually 1.26, but we artificially set it to 1.0 because these
			// options should always be present.
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions:        varOpts,
			DeclTypes:         declTypes,
		},
	)
}

// createVariableOpts creates a slice of EnvOption
// that can be used for creating a CEL env containing variables of declType.
// declType can be nil, in which case the variables will be of DynType.
func createVariableOpts(declType *apiservercel.DeclType, variables ...string) []cel.EnvOption {
	opts := make([]cel.EnvOption, 0, len(variables))
	t := cel.DynType
	if declType != nil {
		t = declType.CelType()
	}
	for _, v := range variables {
		opts = append(opts, cel.Variable(v, t))
	}
	return opts
}

type typeCheckingCompiler struct {
	compositionEnv *plugincel.CompositionEnv
	typeOverwrite  typeOverwrite
}

// CompileCELExpression compiles the given expression.
// The implementation is the same as that of staging/src/k8s.io/apiserver/pkg/admission/plugin/cel/compile.go
// except that:
// - object, oldObject, and params are typed instead of Dyn
// - compiler does not enforce the output type
// - the compiler does not initialize the program
func (c *typeCheckingCompiler) CompileCELExpression(expressionAccessor plugincel.ExpressionAccessor, options plugincel.OptionalVariableDeclarations, mode environment.Type) plugincel.CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) plugincel.CompilationResult {
		return plugincel.CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			ExpressionAccessor: expressionAccessor,
		}
	}
	env, err := c.compositionEnv.Env(mode)
	if err != nil {
		return resultError(fmt.Sprintf("fail to build env: %v", err), apiservercel.ErrorTypeInternal)
	}
	ast, issues := env.Compile(expressionAccessor.GetExpression())
	if issues != nil {
		return resultError(issues.String(), apiservercel.ErrorTypeInvalid)
	}
	// type checker does not require the program, however the type must still be set.
	return plugincel.CompilationResult{
		OutputType: ast.OutputType(),
	}
}

var _ plugincel.Compiler = (*typeCheckingCompiler)(nil)
