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

package cel

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission/plugin/cel/internal/generic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/tools/cache"
)

func (c *celAdmissionController) reconcilePolicyDefinition(namespace, name string, definition PolicyDefinition) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Namespace for policydefinition is empty. Leaving usage here for compatibility
	// with future NamespacedPolicyDefinition
	namespacedName := namespace + "/" + name
	info, ok := c.definitionInfo[namespacedName]
	if !ok {
		info = &definitionInfo{}
		c.definitionInfo[namespacedName] = info
	}

	var paramSource *schema.GroupVersionKind
	if definition != nil {
		paramSource = definition.GetParamSource()
	}

	// If param source has changed, remove definition as dependent of old params
	// If there are no more dependents of old param, stop and clean up controller
	if info.lastReconciledValue != nil && info.lastReconciledValue.GetParamSource() != nil {
		oldParamSource := *info.lastReconciledValue.GetParamSource()

		// If we are:
		//	- switching from having a param to not having a param (includes deletion)
		//	- or from having a param to a different one
		// we remove dependency on the controller.
		if paramSource == nil || *paramSource != oldParamSource {
			if oldParamInfo, ok := c.paramsCRDControllers[oldParamSource]; ok {
				oldParamInfo.dependentDefinitions.Delete(namespacedName)
				if len(oldParamInfo.dependentDefinitions) == 0 {
					oldParamInfo.stop()
					delete(c.paramsCRDControllers, oldParamSource)
				}
			}
		}
	}

	// Reset all previously compiled evaluators in case something relevant in
	// definition has changed.
	for key := range c.definitionsToBindings[namespacedName] {
		bindingInfo := c.bindingInfos[key]
		bindingInfo.evaluator = nil
		c.bindingInfos[key] = bindingInfo
	}

	if definition == nil {
		delete(c.definitionInfo, namespacedName)
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
	paramsGVR, err := c.restMapper.RESTMapping(paramSource.GroupKind(), paramSource.Version)
	if err != nil {
		// Failed to resolve. Return error so we retry again (rate limited)
		// Save a record of this definition with an evaluator that unconditionally
		//
		info.configurationError = fmt.Errorf("failed to find resource for param source: '%v'", paramSource.String())
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
			dependentDefinitions: sets.NewString(namespacedName),
		}

		go informer.Informer().Run(instanceContext.Done())
		go controller.Run(instanceContext)
	}

	return nil
}

func (c *celAdmissionController) reconcilePolicyBinding(namespace, name string, binding PolicyBinding) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Namespace for PolicyBinding is empty. In the future a namespaced binding
	// may be added
	// https://github.com/kubernetes/enhancements/blob/bf5c3c81ea2081d60c1dc7c832faa98479e06209/keps/sig-api-machinery/3488-cel-admission-control/README.md?plain=1#L1042
	namespacedName := namespace + "/" + name
	info, ok := c.bindingInfos[namespacedName]
	if !ok {
		info = &bindingInfo{}
		c.bindingInfos[namespacedName] = info
	}

	oldNamespacedDefinitionName := ""
	if info.lastReconciledValue != nil {
		oldefinitionNamespace, oldefinitionName := info.lastReconciledValue.GetTargetDefinition()
		oldNamespacedDefinitionName = oldefinitionNamespace + "/" + oldefinitionName
	}

	namespacedDefinitionName := ""
	if binding != nil {
		newDefinitionNamespace, newDefinitionName := binding.GetTargetDefinition()
		namespacedDefinitionName = newDefinitionNamespace + "/" + newDefinitionName
	}

	// Remove record of binding from old definition if the referred policy
	// has changed
	if oldNamespacedDefinitionName != namespacedDefinitionName {
		if dependentBindings, ok := c.definitionsToBindings[oldNamespacedDefinitionName]; ok {
			dependentBindings.Delete(namespacedName)

			// if there are no more dependent bindings, remove knowledge of the
			// definition altogether
			if len(dependentBindings) == 0 {
				delete(c.definitionsToBindings, oldNamespacedDefinitionName)
			}
		}
	}

	if binding == nil {
		delete(c.bindingInfos, namespacedName)
		return nil
	}

	// Add record of binding to new definition
	if dependentBindings, ok := c.definitionsToBindings[namespacedDefinitionName]; ok {
		dependentBindings.Insert(namespacedName)
	} else {
		c.definitionsToBindings[namespacedDefinitionName] = sets.NewString(namespacedName)
	}

	// Remove compiled template for old binding
	info.evaluator = nil
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
