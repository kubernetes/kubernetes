/*
Copyright 2024 The Kubernetes Authors.

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
	"context"
	"io"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
)

const (
	// PluginName indicates the name of admission plug-in
	PluginName = "ValidatingAdmissionPolicy"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(configFile io.Reader) (admission.Interface, error) {
		return NewPlugin(configFile), nil
	})
}

// Plugin is an implementation of admission.Interface.
type Policy = v1beta1.ValidatingAdmissionPolicy
type PolicyBinding = v1beta1.ValidatingAdmissionPolicyBinding
type PolicyEvaluator = Validator
type PolicyHook = generic.PolicyHook[*Policy, *PolicyBinding, PolicyEvaluator]

type Plugin struct {
	*generic.Plugin[PolicyHook]
}

var _ admission.Interface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ initializer.WantsFeatures = &Plugin{}

func NewPlugin(_ io.Reader) *Plugin {
	handler := admission.NewHandler(admission.Connect, admission.Create, admission.Delete, admission.Update)

	return &Plugin{
		Plugin: generic.NewPlugin(
			handler,
			func(f informers.SharedInformerFactory, client kubernetes.Interface, dynamicClient dynamic.Interface, restMapper meta.RESTMapper) generic.Source[PolicyHook] {
				return generic.NewPolicySource(
					f.Admissionregistration().V1beta1().ValidatingAdmissionPolicies().Informer(),
					f.Admissionregistration().V1beta1().ValidatingAdmissionPolicyBindings().Informer(),
					NewValidatingAdmissionPolicyAccessor,
					NewValidatingAdmissionPolicyBindingAccessor,
					compilePolicy,
					f,
					dynamicClient,
					restMapper,
				)
			},
			func(a authorizer.Authorizer, m *matching.Matcher) generic.Dispatcher[PolicyHook] {
				return NewDispatcher(a, NewMatcher(m))
			},
		),
	}
}

// Validate makes an admission decision based on the request attributes.
func (a *Plugin) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	return a.Plugin.Dispatch(ctx, attr, o)
}

func (a *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	a.Plugin.SetEnabled(featureGates.Enabled(features.ValidatingAdmissionPolicy))
}

func compilePolicy(policy *Policy) Validator {
	hasParam := false
	if policy.Spec.ParamKind != nil {
		hasParam = true
	}
	optionalVars := cel.OptionalVariableDeclarations{HasParams: hasParam, HasAuthorizer: true}
	expressionOptionalVars := cel.OptionalVariableDeclarations{HasParams: hasParam, HasAuthorizer: false}
	failurePolicy := convertv1beta1FailurePolicyTypeTov1FailurePolicyType(policy.Spec.FailurePolicy)
	var matcher matchconditions.Matcher = nil
	matchConditions := policy.Spec.MatchConditions

	filterCompiler, err := cel.NewCompositedCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
	if err == nil {
		filterCompiler.CompileAndStoreVariables(convertv1beta1Variables(policy.Spec.Variables), optionalVars, environment.StoredExpressions)
	} else {
		//!TODO: return a validator that always fails with internal error?
		utilruntime.HandleError(err)
	}

	if len(matchConditions) > 0 {
		matchExpressionAccessors := make([]cel.ExpressionAccessor, len(matchConditions))
		for i := range matchConditions {
			matchExpressionAccessors[i] = (*matchconditions.MatchCondition)(&matchConditions[i])
		}
		matcher = matchconditions.NewMatcher(filterCompiler.Compile(matchExpressionAccessors, optionalVars, environment.StoredExpressions), failurePolicy, "policy", "validate", policy.Name)
	}
	res := NewValidator(
		filterCompiler.Compile(convertv1beta1Validations(policy.Spec.Validations), optionalVars, environment.StoredExpressions),
		matcher,
		filterCompiler.Compile(convertv1beta1AuditAnnotations(policy.Spec.AuditAnnotations), optionalVars, environment.StoredExpressions),
		filterCompiler.Compile(convertv1beta1MessageExpressions(policy.Spec.Validations), expressionOptionalVars, environment.StoredExpressions),
		failurePolicy,
	)

	return res
}

func convertv1beta1FailurePolicyTypeTov1FailurePolicyType(policyType *v1beta1.FailurePolicyType) *v1.FailurePolicyType {
	if policyType == nil {
		return nil
	}

	var v1FailPolicy v1.FailurePolicyType
	if *policyType == v1beta1.Fail {
		v1FailPolicy = v1.Fail
	} else if *policyType == v1beta1.Ignore {
		v1FailPolicy = v1.Ignore
	}
	return &v1FailPolicy
}

func convertv1beta1Validations(inputValidations []v1beta1.Validation) []cel.ExpressionAccessor {
	celExpressionAccessor := make([]cel.ExpressionAccessor, len(inputValidations))
	for i, validation := range inputValidations {
		validation := ValidationCondition{
			Expression: validation.Expression,
			Message:    validation.Message,
			Reason:     validation.Reason,
		}
		celExpressionAccessor[i] = &validation
	}
	return celExpressionAccessor
}

func convertv1beta1MessageExpressions(inputValidations []v1beta1.Validation) []cel.ExpressionAccessor {
	celExpressionAccessor := make([]cel.ExpressionAccessor, len(inputValidations))
	for i, validation := range inputValidations {
		if validation.MessageExpression != "" {
			condition := MessageExpressionCondition{
				MessageExpression: validation.MessageExpression,
			}
			celExpressionAccessor[i] = &condition
		}
	}
	return celExpressionAccessor
}

func convertv1beta1AuditAnnotations(inputValidations []v1beta1.AuditAnnotation) []cel.ExpressionAccessor {
	celExpressionAccessor := make([]cel.ExpressionAccessor, len(inputValidations))
	for i, validation := range inputValidations {
		validation := AuditAnnotationCondition{
			Key:             validation.Key,
			ValueExpression: validation.ValueExpression,
		}
		celExpressionAccessor[i] = &validation
	}
	return celExpressionAccessor
}

func convertv1beta1Variables(variables []v1beta1.Variable) []cel.NamedExpressionAccessor {
	namedExpressions := make([]cel.NamedExpressionAccessor, len(variables))
	for i, variable := range variables {
		namedExpressions[i] = &Variable{Name: variable.Name, Expression: variable.Expression}
	}
	return namedExpressions
}
