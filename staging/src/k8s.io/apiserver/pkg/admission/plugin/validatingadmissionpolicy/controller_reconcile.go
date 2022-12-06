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
	"time"

	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/tools/cache"
)

func (c *celAdmissionController) reconcilePolicyDefinition(namespace, name string, definition *v1alpha1.ValidatingAdmissionPolicy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Namespace for policydefinition is empty.
	nn := getNamespaceName(namespace, name)
	info, ok := c.definitionInfo[nn]
	if !ok {
		info = &definitionInfo{}
		c.definitionInfo[nn] = info
		// TODO(DangerOnTheRanger): add support for "warn" being a valid enforcementAction
		celmetrics.Metrics.ObserveDefinition(context.TODO(), "active", "deny")
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
		bindingInfo.validator.Store(nil)
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

	// Start watching the param CRD
	if _, ok := c.paramsCRDControllers[*paramSource]; !ok {
		instanceContext, instanceCancel := context.WithCancel(c.runningContext)

		// Watch for new instances of this policy
		informer := dynamicinformer.NewFilteredDynamicInformer(
			c.dynamicClient,
			paramsGVR.Resource,
			corev1.NamespaceAll,
			30*time.Second,
			cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
			nil,
		)

		controller := generic.NewController(
			generic.NewInformer[*unstructured.Unstructured](informer.Informer()),
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

		go informer.Informer().Run(instanceContext.Done())
		go controller.Run(instanceContext)
	}

	return nil
}

func (c *celAdmissionController) reconcilePolicyBinding(namespace, name string, binding *v1alpha1.ValidatingAdmissionPolicyBinding) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Namespace for PolicyBinding is empty. In the future a namespaced binding
	// may be added
	// https://github.com/kubernetes/enhancements/blob/bf5c3c81ea2081d60c1dc7c832faa98479e06209/keps/sig-api-machinery/3488-cel-admission-control/README.md?plain=1#L1042
	nn := getNamespaceName(namespace, name)
	info, ok := c.bindingInfos[nn]
	if !ok {
		info = &bindingInfo{}
		c.bindingInfos[nn] = info
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
	info.validator.Store(nil)
	info.lastReconciledValue = binding
	return nil
}

func (c *celAdmissionController) reconcileParams(namespace, name string, params *unstructured.Unstructured) error {
	// Do nothing.
	// When we add informational type checking we will need to compile in the
	// reconcile loops instead of lazily so we can add compiler errors / type
	// checker errors to the status of the resources.
	return nil
}

func getNamespaceName(namespace, name string) namespacedName {
	return namespacedName{
		namespace: namespace,
		name:      name,
	}
}
