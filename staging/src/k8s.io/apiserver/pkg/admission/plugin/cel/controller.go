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
	"sync"

	"k8s.io/api/admissionregistration/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel/internal/generic"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

var _ CELPolicyEvaluator = &celAdmissionController{}

// celAdmissionController is the top-level controller for admission control using CEL
// it is responsible for watching policy definitions, bindings, and config param CRDs
type celAdmissionController struct {
	// Context under which the controller runs
	runningContext context.Context

	policyDefinitionsController generic.Controller[*v1alpha1.ValidatingAdmissionPolicy]
	policyBindingController     generic.Controller[*v1alpha1.ValidatingAdmissionPolicyBinding]

	// dynamicclient used to create informers to watch the param crd types
	dynamicClient dynamic.Interface
	restMapper    meta.RESTMapper // WantsRESTMapper

	// Provided to the policy's Compile function as an injected dependency to
	// assist with compiling its expressions to CEL
	validatorCompiler ValidatorCompiler

	// Lock which protects:
	//	- definitionInfo
	//  - bindingInfos
	//  - paramCRDControllers
	//  - definitionsToBindings
	// All other fields should be assumed constant
	mutex sync.RWMutex

	// controller and metadata
	paramsCRDControllers map[v1alpha1.ParamKind]*paramInfo

	// Index for each definition namespace/name, contains all binding
	// namespace/names known to exist for that definition
	definitionInfo map[string]*definitionInfo

	// Index for each bindings namespace/name. Contains compiled templates
	// for the binding depending on the policy/param combination.
	bindingInfos map[string]*bindingInfo

	// Map from namespace/name of a definition to a set of namespace/name
	// of bindings which depend on it.
	// All keys must have at least one dependent binding
	// All binding names MUST exist as a key bindingInfos
	definitionsToBindings map[string]sets.String
}

type definitionInfo struct {
	// Error about the state of the definition's configuration and the cluster
	// preventing its enforcement or compilation.
	// Reset every reconciliation
	configurationError error

	// Last value seen by this controller to be used in policy enforcement
	// May not be nil
	lastReconciledValue *v1alpha1.ValidatingAdmissionPolicy
}

type bindingInfo struct {
	// Compiled CEL expression turned into an validator
	validator Validator

	// Last value seen by this controller to be used in policy enforcement
	// May not be nil
	lastReconciledValue *v1alpha1.ValidatingAdmissionPolicyBinding
}

type paramInfo struct {
	// Controller which is watching this param CRD
	controller generic.Controller[*unstructured.Unstructured]

	// Function to call to stop the informer and clean up the controller
	stop func()

	// Policy Definitions which refer to this param CRD
	dependentDefinitions sets.String
}

func NewAdmissionController(
	// Informers
	policyDefinitionsInformer cache.SharedIndexInformer,
	policyBindingInformer cache.SharedIndexInformer,

	// Injected Dependencies
	validatorCompiler ValidatorCompiler,
	restMapper meta.RESTMapper,
	dynamicClient dynamic.Interface,
) CELPolicyEvaluator {
	c := &celAdmissionController{
		definitionInfo:        make(map[string]*definitionInfo),
		bindingInfos:          make(map[string]*bindingInfo),
		paramsCRDControllers:  make(map[v1alpha1.ParamKind]*paramInfo),
		definitionsToBindings: make(map[string]sets.String),
		dynamicClient:         dynamicClient,
		validatorCompiler:     validatorCompiler,
		restMapper:            restMapper,
	}

	c.policyDefinitionsController = generic.NewController(
		generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicy](policyDefinitionsInformer),
		c.reconcilePolicyDefinition,
		generic.ControllerOptions{
			Workers: 1,
			Name:    "cel-policy-definitions",
		},
	)
	c.policyBindingController = generic.NewController(
		generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicyBinding](policyBindingInformer),
		c.reconcilePolicyBinding,
		generic.ControllerOptions{
			Workers: 1,
			Name:    "cel-policy-bindings",
		},
	)
	return c
}

func (c *celAdmissionController) Run(stopCh <-chan struct{}) {
	if c.runningContext != nil {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())

	c.runningContext = ctx
	defer func() {
		c.runningContext = nil
	}()

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

	<-stopCh
	cancel()
	wg.Wait()
}

func (c *celAdmissionController) Validate(
	ctx context.Context,
	a admission.Attributes,
	o admission.ObjectInterfaces,
) (err error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	var deniedDecisions []policyDecisionWithMetadata

	addConfigError := func(err error, definition *v1alpha1.ValidatingAdmissionPolicy, binding *v1alpha1.ValidatingAdmissionPolicyBinding) {
		// we always default the FailurePolicy if it is unset and validate it in API level
		var policy v1alpha1.FailurePolicyType
		if definition.Spec.FailurePolicy == nil {
			policy = v1alpha1.Fail
		} else {
			policy = *definition.Spec.FailurePolicy
		}

		// apply FailurePolicy specified in ValidatingAdmissionPolicy, the default would be Fail
		switch policy {
		case v1alpha1.Ignore:
			if binding == nil {
				klog.Info("ignored ValidatingAdmissionPolicy %w failure, due to FailurePolicy=Ignore. Error: %w", definition.Name, err)
			} else {
				klog.Info("ignored ValidatingAdmissionPolicy %w failure for binding %w, due to FailurePolicy=Ignore. Error: %w", definition.Name, binding.Name, err)
			}
			return
		case v1alpha1.Fail:
			var message string
			if binding == nil {
				message = fmt.Errorf("failed to configure policy: %w", err).Error()
			} else {
				message = fmt.Errorf("failed to configure binding: %w", err).Error()
			}
			deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
				policyDecision: policyDecision{
					kind:    deny,
					message: message,
				},
				definition: definition,
				binding:    binding,
			})
		default:
			utilruntime.HandleError(fmt.Errorf("unrecognized failure policy: '%v'", policy))
		}
	}
	for definitionNamespacedName, definitionInfo := range c.definitionInfo {
		definition := definitionInfo.lastReconciledValue
		matches, err := c.validatorCompiler.DefinitionMatches(definition, a, o)
		if err != nil {
			return err
		}
		if !matches {
			// Policy definition does not match request
			continue
		} else if definitionInfo.configurationError != nil {
			// Configuration error.
			addConfigError(definitionInfo.configurationError, definition, nil)
			continue
		}

		dependentBindings := c.definitionsToBindings[definitionNamespacedName]
		if len(dependentBindings) == 0 {
			continue
		}

		for namespacedBindingName := range dependentBindings {
			// If the key is inside dependentBindings, there is guaranteed to
			// be a bindingInfo for it
			bindingInfo := c.bindingInfos[namespacedBindingName]
			binding := bindingInfo.lastReconciledValue
			matches, err := c.validatorCompiler.BindingMatches(binding, a, o)
			if err != nil {
				return err
			}
			if !matches {
				continue
			}

			var param *unstructured.Unstructured

			// If definition has no paramKind, always provide nil params to
			// evaluator. If binding specifies a params to use they are ignored.
			// Done this way so you can configure params before definition is ready.
			if paramKind := definition.Spec.ParamKind; paramKind != nil {
				paramRef := binding.Spec.ParamRef
				if paramRef == nil {
					// return error if ValidatingAdmissionPolicyBinding is mis-configured
					return fmt.Errorf("ValidatingAdmissionPolicyBinding '%s' requires paramRef since policyName refers to a ValidatingAdmissionPolicy '%s' with paramKind: `%v`",
						binding.Name, definition.Name, paramKind.String())
				}

				// Find the params referred by the binding by looking its name up
				// in our informer for its CRD
				paramInfo, ok := c.paramsCRDControllers[*paramKind]
				if !ok {
					addConfigError(fmt.Errorf("paramKind kind `%v` not known",
						paramKind.String()), definition, binding)
					continue
				}

				if len(paramRef.Namespace) == 0 {
					param, err = paramInfo.controller.Informer().Get(paramRef.Name)
				} else {
					param, err = paramInfo.controller.Informer().Namespaced(paramRef.Namespace).Get(paramRef.Name)
				}

				if err != nil {
					// Apply failure policy
					addConfigError(err, definition, binding)

					if k8serrors.IsNotFound(err) {
						// Param doesnt exist yet?
						// Maybe just have to wait a bit.
						continue
					}

					// There was a bad internal error
					utilruntime.HandleError(err)
					continue
				}
			} else {
				paramRef := binding.Spec.ParamRef
				if paramRef != nil {
					// return error if ValidatingAdmissionPolicyBinding is mis-configured
					return fmt.Errorf("ValidatingAdmissionPolicyBinding '%s' is not allowed to set paramRef to '%s' since policyName refers to a ValidatingAdmissionPolicy '%s' which has no paramKind",
						binding.Name, paramRef.String(), definition.Name)
				}
			}

			if bindingInfo.validator == nil {
				// Compile policy definition using binding
				bindingInfo.validator = c.validatorCompiler.Compile(definition)
				if err != nil {
					// compilation error. Apply failure policy
					wrappedError := fmt.Errorf("failed to compile CEL expression: %w", err)
					addConfigError(wrappedError, definition, binding)
					continue
				}
				c.bindingInfos[namespacedBindingName] = bindingInfo
			}

			decisions, err := bindingInfo.validator.Validate(a, o, param)
			if err != nil {
				// runtime error. Apply failure policy
				wrappedError := fmt.Errorf("failed to evaluate CEL expression: %w", err)
				addConfigError(wrappedError, definition, binding)
				continue
			}

			for _, decision := range decisions {
				switch decision.kind {
				case admit:
					if len(decision.message) != 0 {
						klog.Info("Ignored ValidatingAdmissionPolicy %w failure for binding %w, due to FailurePolicy=Ignore. Error: %w", definition.Name, binding.Name, decision.message)
					}
				case deny:
					deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
						definition:     definition,
						binding:        binding,
						policyDecision: decision,
					})
				default:
					// unrecognized decision. ignore
				}
			}
		}
	}

	if len(deniedDecisions) > 0 {
		return &policyError{
			deniedDecisions: deniedDecisions,
		}
	}

	return nil
}

func (c *celAdmissionController) HasSynced() bool {
	return c.policyBindingController.HasSynced() &&
		c.policyDefinitionsController.HasSynced()
}
