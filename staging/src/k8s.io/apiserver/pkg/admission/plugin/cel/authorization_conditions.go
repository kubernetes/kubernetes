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

package cel

import (
	"context"

	celgo "github.com/google/cel-go/cel"
	celtypes "github.com/google/cel-go/common/types"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	namespacematcher "k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
)

const (
	ConditionTypeAuthorizationCEL = "k8s.io/authorization-cel"
)

func NewAuthorizationConditionsEvaluator(authorizer authorizer.Authorizer, informerFactory informers.SharedInformerFactory, client kubernetes.Interface, runtimeCELCostBudget int64) *AuthorizationConditionsEvaluator {
	var nsmatcher *namespacematcher.Matcher
	if informerFactory != nil && client != nil {
		nsmatcher = &namespacematcher.Matcher{
			NamespaceLister: informerFactory.Core().V1().Namespaces().Lister(),
			Client:          client,
		}
	}

	compositionEnvTemplateWithStrictCost := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())

	return &AuthorizationConditionsEvaluator{
		compiler: NewCompiler(compositionEnvTemplateWithStrictCost),
		// TODO: Add unit test for using the authorizer
		optionalVarsDecls:    OptionalVariableDeclarations{HasParams: false, HasAuthorizer: authorizer != nil},
		optionalVars:         OptionalVariableBindings{VersionedParams: nil, Authorizer: authorizer},
		nsmatcher:            nsmatcher,
		runtimeCELCostBudget: runtimeCELCostBudget,
	}
}

type AuthorizationConditionsEvaluator struct {
	compiler             Compiler
	optionalVarsDecls    OptionalVariableDeclarations
	optionalVars         OptionalVariableBindings
	nsmatcher            *namespacematcher.Matcher
	runtimeCELCostBudget int64
}

// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (e *AuthorizationConditionsEvaluator) EvaluateCondition(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
	// First, check all the essentials, that the condition type is correct, and there is some admission data to evaluate
	if condition == nil {
		return authorizer.ConditionsEvaluationResultUnevaluatable()
	}
	if data.AdmissionControl == nil {
		return authorizer.ConditionsEvaluationResultUnevaluatable()
	}
	if condition.GetType() != ConditionTypeAuthorizationCEL {
		return authorizer.ConditionsEvaluationResultUnevaluatable()
	}
	// It's generally valid for condition strings to be empty, if the semantics of the condition is
	// clear from its (required) ID. However, the CEL evaluator cannot evaluate empty condition strings.
	condStr := condition.GetCondition()
	if len(condStr) == 0 {
		return authorizer.ConditionsEvaluationResultUnevaluatable()
	}

	versionedAttrs := admission.VersionedAttributes{
		Attributes: admission.NewAttributesRecord(
			data.AdmissionControl.GetObject(),
			data.AdmissionControl.GetOldObject(),
			data.AdmissionControl.GetKind(),
			data.AdmissionControl.GetNamespace(),
			data.AdmissionControl.GetName(),
			data.AdmissionControl.GetResource(),
			data.AdmissionControl.GetSubresource(),
			data.AdmissionControl.GetOperation(),
			data.AdmissionControl.GetOperationOptions(),
			data.AdmissionControl.IsDryRun(),
			data.AdmissionControl.GetUserInfo(),
		),
		VersionedObject:    data.AdmissionControl.GetObject(),
		VersionedOldObject: data.AdmissionControl.GetOldObject(),
		VersionedKind:      data.AdmissionControl.GetKind(),
	}

	admissionRequest := CreateAdmissionRequest(&versionedAttrs, metav1.GroupVersionResource(versionedAttrs.GetResource()), metav1.GroupVersionKind(versionedAttrs.GetKind()))

	// TODO(luxas): Would it be faster to compile all deny conditions at a time, then all no opinion, etc?
	// Most likely, as the same activation variable could be re-used in this case.
	compiledExpr := e.compiler.CompileCELExpression(validationCondition(condStr), e.optionalVarsDecls, environment.StoredExpressions)

	ns, err := e.getNamespace(ctx, &versionedAttrs)
	if err != nil {
		return authorizer.ConditionEvaluationResultError(err)
	}

	// if this activation supports composition, we will need the compositionCtx. It may be nil.
	compositionCtx, _ := ctx.(CompositionContext)

	activation, err := newActivation(compositionCtx, &versionedAttrs, admissionRequest, e.optionalVars, ns)
	if err != nil {
		return authorizer.ConditionEvaluationResultError(err)
	}

	evalResult, _, err := activation.Evaluate(ctx, compositionCtx, compiledExpr, e.runtimeCELCostBudget)
	if err != nil {
		return authorizer.ConditionEvaluationResultError(err)
	}

	// TODO(luxas): Do we trust the authorizer to give reasonable-length conditions?
	// Can a webhook authorizer today block a request for e.g. 60s?
	if evalResult.Error != nil {
		return authorizer.ConditionEvaluationResultError(evalResult.Error)
	}

	return authorizer.ConditionEvaluationResultBoolean(evalResult.EvalResult == celtypes.True)
}

// getNamespace exactly mirrors the similar logic in VAP
func (e *AuthorizationConditionsEvaluator) getNamespace(ctx context.Context, attrs admission.Attributes) (*v1.Namespace, error) {
	// If nsmatcher isn't available (due to no client), we just don't populate the namespaceObject variable, even though
	// the object would be namespaced.
	if e.nsmatcher == nil {
		return nil, nil
	}

	var namespace *v1.Namespace
	var err error
	namespaceName := attrs.GetNamespace()

	// Special case, the namespace object has the namespace of itself (maybe a bug).
	// unset it if the incoming object is a namespace
	if gvk := attrs.GetKind(); gvk.Kind == "Namespace" && gvk.Version == "v1" && gvk.Group == "" {
		namespaceName = ""
	}

	// if it is cluster scoped, namespaceName will be empty
	// Otherwise, get the Namespace resource.
	if namespaceName != "" {
		namespace, err = e.nsmatcher.GetNamespace(ctx, namespaceName)
		if err != nil {
			return nil, err
		}
	}

	return CreateNamespaceObject(namespace), nil
}

type validationCondition string

func (v validationCondition) GetExpression() string {
	return string(v)
}

func (v validationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}
