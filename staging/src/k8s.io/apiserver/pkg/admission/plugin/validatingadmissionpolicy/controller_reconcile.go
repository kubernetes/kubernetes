/*
Copyright 2022 The Kubernetes Authors.

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

package validatingadmissionpolicy

import (
	"context"
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

type policyController struct {
	once                        sync.Once
	context                     context.Context
	dynamicClient               dynamic.Interface
	informerFactory             informers.SharedInformerFactory
	restMapper                  meta.RESTMapper
	policyDefinitionsController generic.Controller[*v1beta1.ValidatingAdmissionPolicy]
	policyBindingController     generic.Controller[*v1beta1.ValidatingAdmissionPolicyBinding]

	// Provided to the policy's Compile function as an injected dependency to
	// assist with compiling its expressions to CEL
	// pass nil to create filter compiler in demand
	filterCompiler cel.FilterCompiler

	matcher Matcher

	newValidator

	client kubernetes.Interface
	// Lock which protects
	// All Below fields
	// All above fields should be assumed constant
	mutex sync.RWMutex

	cachedPolicies []policyData

	// controller and metadata
	paramsCRDControllers map[v1beta1.ParamKind]*paramInfo

	// Index for each definition namespace/name, contains all binding
	// namespace/names known to exist for that definition
	definitionInfo map[namespacedName]*definitionInfo

	// Index for each bindings namespace/name. Contains compiled templates
	// for the binding depending on the policy/param combination.
	bindingInfos map[namespacedName]*bindingInfo

	// Map from namespace/name of a definition to a set of namespace/name
	// of bindings which depend on it.
	// All keys must have at least one dependent binding
	// All binding names MUST exist as a key bindingInfos
	definitionsToBindings map[namespacedName]sets.Set[namespacedName]
}

type newValidator func(validationFilter cel.Filter, celMatcher matchconditions.Matcher, auditAnnotationFilter, messageFilter cel.Filter, failurePolicy *v1.FailurePolicyType) Validator

func newPolicyController(
	restMapper meta.RESTMapper,
	client kubernetes.Interface,
	dynamicClient dynamic.Interface,
	informerFactory informers.SharedInformerFactory,
	filterCompiler cel.FilterCompiler,
	matcher Matcher,
	policiesInformer generic.Informer[*v1beta1.ValidatingAdmissionPolicy],
	bindingsInformer generic.Informer[*v1beta1.ValidatingAdmissionPolicyBinding],
) *policyController {
	res := &policyController{}
	*res = policyController{
		filterCompiler:        filterCompiler,
		definitionInfo:        make(map[namespacedName]*definitionInfo),
		bindingInfos:          make(map[namespacedName]*bindingInfo),
		paramsCRDControllers:  make(map[v1beta1.ParamKind]*paramInfo),
		definitionsToBindings: make(map[namespacedName]sets.Set[namespacedName]),
		matcher:               matcher,
		newValidator:          NewValidator,
		policyDefinitionsController: generic.NewController(
			policiesInformer,
			res.reconcilePolicyDefinition,
			generic.ControllerOptions{
				Workers: 1,
				Name:    "cel-policy-definitions",
			},
		),
		policyBindingController: generic.NewController(
			bindingsInformer,
			res.reconcilePolicyBinding,
			generic.ControllerOptions{
				Workers: 1,
				Name:    "cel-policy-bindings",
			},
		),
		restMapper:      restMapper,
		dynamicClient:   dynamicClient,
		informerFactory: informerFactory,
		client:          client,
	}
	return res
}

func (c *policyController) Run(ctx context.Context) {
	// Only support being run once
	c.once.Do(func() {
		c.context = ctx

		wg := sync.WaitGroup{}

		wg.Add(1)
		go func() {
			defer wg.Done()
			c.policyDefinitionsController.Run(ctx)
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			c.policyBindingController.Run(ctx)
		}()

		<-ctx.Done()
		wg.Wait()
	})
}

func (c *policyController) HasSynced() bool {
	return c.policyDefinitionsController.HasSynced() && c.policyBindingController.HasSynced()
}

func (c *policyController) reconcilePolicyDefinition(namespace, name string, definition *v1beta1.ValidatingAdmissionPolicy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	err := c.reconcilePolicyDefinitionSpec(namespace, name, definition)
	return err
}

func (c *policyController) reconcilePolicyDefinitionSpec(namespace, name string, definition *v1beta1.ValidatingAdmissionPolicy) error {
	c.cachedPolicies = nil // invalidate cachedPolicies

	// Namespace for policydefinition is empty.
	nn := getNamespaceName(namespace, name)
	info, ok := c.definitionInfo[nn]
	if !ok {
		info = &definitionInfo{}
		c.definitionInfo[nn] = info
		// TODO(DangerOnTheRanger): add support for "warn" being a valid enforcementAction
		celmetrics.Metrics.ObserveDefinition(context.TODO(), "active", "deny")
	}

	// Skip reconcile if the spec of the definition is unchanged and had a
	// successful previous sync
	if info.configurationError == nil && info.lastReconciledValue != nil && definition != nil &&
		apiequality.Semantic.DeepEqual(info.lastReconciledValue.Spec, definition.Spec) {
		return nil
	}

	var paramSource *v1beta1.ParamKind
	if definition != nil {
		paramSource = definition.Spec.ParamKind
	}

	// If param source has changed, remove definition as dependent of old params
	// If there are no more dependents of old param, stop and clean up controller
	if info.lastReconciledValue != nil && info.lastReconciledValue.Spec.ParamKind != nil {
		oldParamSource := *info.lastReconciledValue.Spec.ParamKind

		// If we are:
		//	- switching from having a param to not having a param (includes deletion)
		//	- or from having a param to a different one
		// we remove dependency on the controller.
		if paramSource == nil || *paramSource != oldParamSource {
			if oldParamInfo, ok := c.paramsCRDControllers[oldParamSource]; ok {
				oldParamInfo.dependentDefinitions.Delete(nn)
				if len(oldParamInfo.dependentDefinitions) == 0 {
					oldParamInfo.stop()
					delete(c.paramsCRDControllers, oldParamSource)
				}
			}
		}
	}

	// Reset all previously compiled evaluators in case something relevant in
	// definition has changed.
	for key := range c.definitionsToBindings[nn] {
		bindingInfo := c.bindingInfos[key]
		bindingInfo.validator = nil
		c.bindingInfos[key] = bindingInfo
	}

	if definition == nil {
		delete(c.definitionInfo, nn)
		return nil
	}

	// Update definition info
	info.lastReconciledValue = definition
	info.configurationError = nil

	if paramSource == nil {
		// Skip setting up controller for empty param type
		return nil
	}
	// find GVR for params
	// Parse param source into a GVK

	paramSourceGV, err := schema.ParseGroupVersion(paramSource.APIVersion)
	if err != nil {
		// Failed to resolve. Return error so we retry again (rate limited)
		// Save a record of this definition with an evaluator that unconditionally
		info.configurationError = fmt.Errorf("failed to parse apiVersion of paramKind '%v' with error: %w", paramSource.String(), err)

		// Return nil, since this error cannot be resolved by waiting more time
		return nil
	}

	paramsGVR, err := c.restMapper.RESTMapping(schema.GroupKind{
		Group: paramSourceGV.Group,
		Kind:  paramSource.Kind,
	}, paramSourceGV.Version)

	if err != nil {
		// Failed to resolve. Return error so we retry again (rate limited)
		// Save a record of this definition with an evaluator that unconditionally
		//
		info.configurationError = fmt.Errorf("failed to find resource referenced by paramKind: '%v'", paramSourceGV.WithKind(paramSource.Kind))
		return info.configurationError
	}

	paramInfo := c.ensureParamInfo(paramSource, paramsGVR)
	paramInfo.dependentDefinitions.Insert(nn)

	return nil
}

// Ensures that there is an informer started for the given GVK to be used as a
// param
func (c *policyController) ensureParamInfo(paramSource *v1beta1.ParamKind, mapping *meta.RESTMapping) *paramInfo {
	if info, ok := c.paramsCRDControllers[*paramSource]; ok {
		return info
	}

	// We are not watching this param. Start an informer for it.
	instanceContext, instanceCancel := context.WithCancel(c.context)

	var informer cache.SharedIndexInformer

	// Try to see if our provided informer factory has an informer for this type.
	// We assume the informer is already started, and starts all types associated
	// with it.
	if genericInformer, err := c.informerFactory.ForResource(mapping.Resource); err == nil {
		informer = genericInformer.Informer()

		// Ensure the informer is started
		// Use policyController's context rather than the instance context.
		// PolicyController context is expected to last until app shutdown
		// This is due to behavior of informerFactory which would cause the
		// informer to stop running once the context is cancelled, and
		// never started again.
		c.informerFactory.Start(c.context.Done())
	} else {
		// Dynamic JSON informer fallback.
		// Cannot use shared dynamic informer since it would be impossible
		// to clean CRD informers properly with multiple dependents
		// (cannot start ahead of time, and cannot track dependencies via stopCh)
		informer = dynamicinformer.NewFilteredDynamicInformer(
			c.dynamicClient,
			mapping.Resource,
			corev1.NamespaceAll,
			// Use same interval as is used for k8s typed sharedInformerFactory
			// https://github.com/kubernetes/kubernetes/blob/7e0923899fed622efbc8679cca6b000d43633e38/cmd/kube-apiserver/app/server.go#L430
			10*time.Minute,
			cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
			nil,
		).Informer()
		go informer.Run(instanceContext.Done())
	}

	controller := generic.NewController(
		generic.NewInformer[runtime.Object](informer),
		c.reconcileParams,
		generic.ControllerOptions{
			Workers: 1,
			Name:    paramSource.String() + "-controller",
		},
	)

	ret := &paramInfo{
		controller:           controller,
		stop:                 instanceCancel,
		scope:                mapping.Scope,
		dependentDefinitions: sets.New[namespacedName](),
	}
	c.paramsCRDControllers[*paramSource] = ret

	go controller.Run(instanceContext)
	return ret

}

func (c *policyController) reconcilePolicyBinding(namespace, name string, binding *v1beta1.ValidatingAdmissionPolicyBinding) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cachedPolicies = nil // invalidate cachedPolicies

	// Namespace for PolicyBinding is empty. In the future a namespaced binding
	// may be added
	// https://github.com/kubernetes/enhancements/blob/bf5c3c81ea2081d60c1dc7c832faa98479e06209/keps/sig-api-machinery/3488-cel-admission-control/README.md?plain=1#L1042
	nn := getNamespaceName(namespace, name)
	info, ok := c.bindingInfos[nn]
	if !ok {
		info = &bindingInfo{}
		c.bindingInfos[nn] = info
	}

	// Skip if the spec of the binding is unchanged.
	if info.lastReconciledValue != nil && binding != nil &&
		apiequality.Semantic.DeepEqual(info.lastReconciledValue.Spec, binding.Spec) {
		return nil
	}

	var oldNamespacedDefinitionName namespacedName
	if info.lastReconciledValue != nil {
		// All validating policies are cluster-scoped so have empty namespace
		oldNamespacedDefinitionName = getNamespaceName("", info.lastReconciledValue.Spec.PolicyName)
	}

	var namespacedDefinitionName namespacedName
	if binding != nil {
		// All validating policies are cluster-scoped so have empty namespace
		namespacedDefinitionName = getNamespaceName("", binding.Spec.PolicyName)
	}

	// Remove record of binding from old definition if the referred policy
	// has changed
	if oldNamespacedDefinitionName != namespacedDefinitionName {
		if dependentBindings, ok := c.definitionsToBindings[oldNamespacedDefinitionName]; ok {
			dependentBindings.Delete(nn)

			// if there are no more dependent bindings, remove knowledge of the
			// definition altogether
			if len(dependentBindings) == 0 {
				delete(c.definitionsToBindings, oldNamespacedDefinitionName)
			}
		}
	}

	if binding == nil {
		delete(c.bindingInfos, nn)
		return nil
	}

	// Add record of binding to new definition
	if dependentBindings, ok := c.definitionsToBindings[namespacedDefinitionName]; ok {
		dependentBindings.Insert(nn)
	} else {
		c.definitionsToBindings[namespacedDefinitionName] = sets.New(nn)
	}

	// Remove compiled template for old binding
	info.validator = nil
	info.lastReconciledValue = binding
	return nil
}

func (c *policyController) reconcileParams(namespace, name string, params runtime.Object) error {
	// Do nothing.
	// When we add informational type checking we will need to compile in the
	// reconcile loops instead of lazily so we can add compiler errors / type
	// checker errors to the status of the resources.
	return nil
}

// Fetches the latest set of policy data or recalculates it if it has changed
// since it was last fetched
func (c *policyController) latestPolicyData() []policyData {
	existing := func() []policyData {
		c.mutex.RLock()
		defer c.mutex.RUnlock()

		return c.cachedPolicies
	}()

	if existing != nil {
		return existing
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	var res []policyData
	for definitionNN, definitionInfo := range c.definitionInfo {
		var bindingInfos []bindingInfo
		for bindingNN := range c.definitionsToBindings[definitionNN] {
			bindingInfo := c.bindingInfos[bindingNN]
			if bindingInfo.validator == nil && definitionInfo.configurationError == nil {
				hasParam := false
				if definitionInfo.lastReconciledValue.Spec.ParamKind != nil {
					hasParam = true
				}
				optionalVars := cel.OptionalVariableDeclarations{HasParams: hasParam, HasAuthorizer: true}
				expressionOptionalVars := cel.OptionalVariableDeclarations{HasParams: hasParam, HasAuthorizer: false}
				failurePolicy := convertv1beta1FailurePolicyTypeTov1FailurePolicyType(definitionInfo.lastReconciledValue.Spec.FailurePolicy)
				var matcher matchconditions.Matcher = nil
				matchConditions := definitionInfo.lastReconciledValue.Spec.MatchConditions

				filterCompiler := c.filterCompiler
				if filterCompiler == nil {
					compositedCompiler, err := cel.NewCompositedCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
					if err == nil {
						filterCompiler = compositedCompiler
						compositedCompiler.CompileAndStoreVariables(convertv1beta1Variables(definitionInfo.lastReconciledValue.Spec.Variables), optionalVars, environment.StoredExpressions)
					} else {
						utilruntime.HandleError(err)
					}
				}
				if len(matchConditions) > 0 {
					matchExpressionAccessors := make([]cel.ExpressionAccessor, len(matchConditions))
					for i := range matchConditions {
						matchExpressionAccessors[i] = (*matchconditions.MatchCondition)(&matchConditions[i])
					}
					matcher = matchconditions.NewMatcher(filterCompiler.Compile(matchExpressionAccessors, optionalVars, environment.StoredExpressions), failurePolicy, "policy", "validate", definitionInfo.lastReconciledValue.Name)
				}
				bindingInfo.validator = c.newValidator(
					filterCompiler.Compile(convertv1beta1Validations(definitionInfo.lastReconciledValue.Spec.Validations), optionalVars, environment.StoredExpressions),
					matcher,
					filterCompiler.Compile(convertv1beta1AuditAnnotations(definitionInfo.lastReconciledValue.Spec.AuditAnnotations), optionalVars, environment.StoredExpressions),
					filterCompiler.Compile(convertv1beta1MessageExpressions(definitionInfo.lastReconciledValue.Spec.Validations), expressionOptionalVars, environment.StoredExpressions),
					failurePolicy,
				)
			}
			bindingInfos = append(bindingInfos, *bindingInfo)
		}

		var pInfo paramInfo
		if paramKind := definitionInfo.lastReconciledValue.Spec.ParamKind; paramKind != nil {
			if info, ok := c.paramsCRDControllers[*paramKind]; ok {
				pInfo = *info
			}
		}

		res = append(res, policyData{
			definitionInfo: *definitionInfo,
			paramInfo:      pInfo,
			bindings:       bindingInfos,
		})
	}

	c.cachedPolicies = res
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

func getNamespaceName(namespace, name string) namespacedName {
	return namespacedName{
		namespace: namespace,
		name:      name,
	}
}
