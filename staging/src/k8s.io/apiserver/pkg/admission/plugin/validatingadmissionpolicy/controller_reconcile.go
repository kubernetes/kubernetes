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
	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	k8sscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
)

type policyController struct {
	once                        sync.Once
	context                     context.Context
	dynamicClient               dynamic.Interface
	restMapper                  meta.RESTMapper
	policyDefinitionsController generic.Controller[*v1alpha1.ValidatingAdmissionPolicy]
	policyBindingController     generic.Controller[*v1alpha1.ValidatingAdmissionPolicyBinding]

	// Provided to the policy's Compile function as an injected dependency to
	// assist with compiling its expressions to CEL
	filterCompiler cel.FilterCompiler

	matcher Matcher

	newValidator

	// The TypeCheck checks the policy's expressions for type errors.
	// Type of params is defined in policy.Spec.ParamsKind
	// Types of object are calculated from policy.Spec.MatchingConstraints
	typeChecker *TypeChecker

	// Lock which protects:
	//  - cachedPolicies
	//  - paramCRDControllers
	//  - definitionInfo
	//  - bindingInfos
	//  - definitionsToBindings
	// All other fields should be assumed constant
	mutex sync.RWMutex

	cachedPolicies []policyData

	// controller and metadata
	paramsCRDControllers map[v1alpha1.ParamKind]*paramInfo

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

	client kubernetes.Interface

	authz authorizer.Authorizer
}

type newValidator func(validationFilter cel.Filter, celMatcher matchconditions.Matcher, auditAnnotationFilter, messageFilter cel.Filter, failurePolicy *v1.FailurePolicyType, authorizer authorizer.Authorizer) Validator

func newPolicyController(
	restMapper meta.RESTMapper,
	client kubernetes.Interface,
	dynamicClient dynamic.Interface,
	typeChecker *TypeChecker,
	filterCompiler cel.FilterCompiler,
	matcher Matcher,
	policiesInformer generic.Informer[*v1alpha1.ValidatingAdmissionPolicy],
	bindingsInformer generic.Informer[*v1alpha1.ValidatingAdmissionPolicyBinding],
	authz authorizer.Authorizer,
) *policyController {
	res := &policyController{}
	*res = policyController{
		filterCompiler:        filterCompiler,
		typeChecker:           typeChecker,
		definitionInfo:        make(map[namespacedName]*definitionInfo),
		bindingInfos:          make(map[namespacedName]*bindingInfo),
		paramsCRDControllers:  make(map[v1alpha1.ParamKind]*paramInfo),
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
		restMapper:    restMapper,
		dynamicClient: dynamicClient,
		client:        client,
		authz:         authz,
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

func (c *policyController) reconcilePolicyDefinition(namespace, name string, definition *v1alpha1.ValidatingAdmissionPolicy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	err := c.reconcilePolicyDefinitionSpec(namespace, name, definition)
	if err != nil {
		return err
	}
	if c.typeChecker != nil {
		err = c.reconcilePolicyStatus(namespace, name, definition)
	}
	return err
}

func (c *policyController) reconcilePolicyDefinitionSpec(namespace, name string, definition *v1alpha1.ValidatingAdmissionPolicy) error {
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

	// Skip reconcile if the spec of the definition is unchanged
	if info.lastReconciledValue != nil && definition != nil &&
		apiequality.Semantic.DeepEqual(info.lastReconciledValue.Spec, definition.Spec) {
		return nil
	}

	var paramSource *v1alpha1.ParamKind
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

	if info, ok := c.paramsCRDControllers[*paramSource]; ok {
		// If a param controller is already active for this paramsource, make
		// sure it is tracking this policy's dependency upon it
		info.dependentDefinitions.Insert(nn)

	} else {
		instanceContext, instanceCancel := context.WithCancel(c.context)

		var informer cache.SharedIndexInformer

		// Informer Factory is optional
		if c.client != nil {
			// Create temporary informer factory
			// Cannot use the k8s shared informer factory for dynamic params informer.
			// Would leak unnecessary informers when we are done since we would have to
			// call informerFactory.Start() with a longer-lived stopCh than necessary.
			// SharedInformerFactory does not support temporary usage.
			dynamicFactory := informers.NewSharedInformerFactory(c.client, 10*time.Minute)

			// Look for a typed informer. If it does not exist
			genericInformer, err := dynamicFactory.ForResource(paramsGVR.Resource)

			// Ignore error. We fallback to dynamic informer if there is no
			// typed informer
			if err != nil {
				informer = nil
			} else {
				informer = genericInformer.Informer()

				// Set transformer on the informer to workaround inconsistency
				// where typed objects have TypeMeta wiped out but dynamic
				// objects keep kind/apiVersion fields
				informer.SetTransform(func(i interface{}) (interface{}, error) {
					// Ensure param is populated with its GVK for consistency
					// (CRD dynamic informer always returns objects with kind/apiversion,
					// but native types do not include populated TypeMeta.
					if param := i.(runtime.Object); param != nil {
						if param.GetObjectKind().GroupVersionKind().Empty() {
							// https://github.com/kubernetes/client-go/issues/413#issue-324586398
							gvks, _, _ := k8sscheme.Scheme.ObjectKinds(param)
							for _, gvk := range gvks {
								if len(gvk.Kind) == 0 {
									continue
								}
								if len(gvk.Version) == 0 || gvk.Version == runtime.APIVersionInternal {
									continue
								}
								param.GetObjectKind().SetGroupVersionKind(gvk)
								break
							}
						}
					}

					return i, nil
				})
			}
		}

		if informer == nil {
			// Dynamic JSON informer fallback.
			// Cannot use shared dynamic informer since it would be impossible
			// to clean CRD informers properly with multiple dependents
			// (cannot start ahead of time, and cannot track dependencies via stopCh)
			informer = dynamicinformer.NewFilteredDynamicInformer(
				c.dynamicClient,
				paramsGVR.Resource,
				corev1.NamespaceAll,
				// Use same interval as is used for k8s typed sharedInformerFactory
				// https://github.com/kubernetes/kubernetes/blob/7e0923899fed622efbc8679cca6b000d43633e38/cmd/kube-apiserver/app/server.go#L430
				10*time.Minute,
				cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
				nil,
			).Informer()
		}

		controller := generic.NewController(
			generic.NewInformer[runtime.Object](informer),
			c.reconcileParams,
			generic.ControllerOptions{
				Workers: 1,
				Name:    paramSource.String() + "-controller",
			},
		)

		c.paramsCRDControllers[*paramSource] = &paramInfo{
			controller:           controller,
			stop:                 instanceCancel,
			dependentDefinitions: sets.New(nn),
		}

		go controller.Run(instanceContext)
		go informer.Run(instanceContext.Done())
	}

	return nil
}

func (c *policyController) reconcilePolicyBinding(namespace, name string, binding *v1alpha1.ValidatingAdmissionPolicyBinding) error {
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

func (c *policyController) reconcilePolicyStatus(namespace, name string, definition *v1alpha1.ValidatingAdmissionPolicy) error {
	if definition != nil && definition.Status.ObservedGeneration < definition.Generation {
		st := c.calculatePolicyStatus(definition)
		newDefinition := definition.DeepCopy()
		newDefinition.Status = *st
		_, err := c.client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().UpdateStatus(c.context, newDefinition, metav1.UpdateOptions{})
		if err != nil {
			// ignore error when the controller is not able to
			// mutate the definition, and to avoid infinite requeue.
			utilruntime.HandleError(err)
		}
	}
	return nil
}

func (c *policyController) calculatePolicyStatus(definition *v1alpha1.ValidatingAdmissionPolicy) *v1alpha1.ValidatingAdmissionPolicyStatus {
	expressionWarnings := c.typeChecker.Check(definition)
	// modifying a deepcopy of the original status, preserving unrelated existing data
	status := definition.Status.DeepCopy()
	status.ObservedGeneration = definition.Generation
	status.TypeChecking = &v1alpha1.TypeChecking{ExpressionWarnings: expressionWarnings}
	return status
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
				failurePolicy := convertv1alpha1FailurePolicyTypeTov1FailurePolicyType(definitionInfo.lastReconciledValue.Spec.FailurePolicy)
				var matcher matchconditions.Matcher = nil
				matchConditions := definitionInfo.lastReconciledValue.Spec.MatchConditions
				if len(matchConditions) > 0 {
					matchExpressionAccessors := make([]cel.ExpressionAccessor, len(matchConditions))
					for i := range matchConditions {
						matchExpressionAccessors[i] = (*matchconditions.MatchCondition)(&matchConditions[i])
					}
					matcher = matchconditions.NewMatcher(c.filterCompiler.Compile(matchExpressionAccessors, optionalVars, celconfig.PerCallLimit), c.authz, failurePolicy, "validatingadmissionpolicy", definitionInfo.lastReconciledValue.Name)
				}
				bindingInfo.validator = c.newValidator(
					c.filterCompiler.Compile(convertv1alpha1Validations(definitionInfo.lastReconciledValue.Spec.Validations), optionalVars, celconfig.PerCallLimit),
					matcher,
					c.filterCompiler.Compile(convertv1alpha1AuditAnnotations(definitionInfo.lastReconciledValue.Spec.AuditAnnotations), optionalVars, celconfig.PerCallLimit),
					c.filterCompiler.Compile(convertV1Alpha1MessageExpressions(definitionInfo.lastReconciledValue.Spec.Validations), expressionOptionalVars, celconfig.PerCallLimit),
					failurePolicy,
					c.authz,
				)
			}
			bindingInfos = append(bindingInfos, *bindingInfo)
		}

		var paramController generic.Controller[runtime.Object]
		if paramKind := definitionInfo.lastReconciledValue.Spec.ParamKind; paramKind != nil {
			if info, ok := c.paramsCRDControllers[*paramKind]; ok {
				paramController = info.controller
			}
		}

		res = append(res, policyData{
			definitionInfo:  *definitionInfo,
			paramController: paramController,
			bindings:        bindingInfos,
		})
	}

	c.cachedPolicies = res
	return res
}

func convertv1alpha1FailurePolicyTypeTov1FailurePolicyType(policyType *v1alpha1.FailurePolicyType) *v1.FailurePolicyType {
	if policyType == nil {
		return nil
	}

	var v1FailPolicy v1.FailurePolicyType
	if *policyType == v1alpha1.Fail {
		v1FailPolicy = v1.Fail
	} else if *policyType == v1alpha1.Ignore {
		v1FailPolicy = v1.Ignore
	}
	return &v1FailPolicy
}

func convertv1alpha1Validations(inputValidations []v1alpha1.Validation) []cel.ExpressionAccessor {
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

func convertV1Alpha1MessageExpressions(inputValidations []v1alpha1.Validation) []cel.ExpressionAccessor {
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

func convertv1alpha1AuditAnnotations(inputValidations []v1alpha1.AuditAnnotation) []cel.ExpressionAccessor {
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

func getNamespaceName(namespace, name string) namespacedName {
	return namespacedName{
		namespace: namespace,
		name:      name,
	}
}
