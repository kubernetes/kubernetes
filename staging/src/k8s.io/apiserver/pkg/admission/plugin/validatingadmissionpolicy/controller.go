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
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/matching"

	"k8s.io/api/admissionregistration/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
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
	restMapper    meta.RESTMapper

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

// namespaceName is used as a key in definitionInfo and bindingInfos
type namespacedName struct {
	namespace, name string
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
	validator atomic.Pointer[Validator]

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
	dependentDefinitions sets.Set[namespacedName]
}

func NewAdmissionController(
	// Injected Dependencies
	informerFactory informers.SharedInformerFactory,
	client kubernetes.Interface,
	restMapper meta.RESTMapper,
	dynamicClient dynamic.Interface,
) CELPolicyEvaluator {
	matcher := matching.NewMatcher(informerFactory.Core().V1().Namespaces().Lister(), client)
	validatorCompiler := &CELValidatorCompiler{
		Matcher: matcher,
	}
	c := &celAdmissionController{
		definitionInfo:        make(map[namespacedName]*definitionInfo),
		bindingInfos:          make(map[namespacedName]*bindingInfo),
		paramsCRDControllers:  make(map[v1alpha1.ParamKind]*paramInfo),
		definitionsToBindings: make(map[namespacedName]sets.Set[namespacedName]),
		dynamicClient:         dynamicClient,
		validatorCompiler:     validatorCompiler,
		restMapper:            restMapper,
	}

	c.policyDefinitionsController = generic.NewController(
		generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicy](
			informerFactory.Admissionregistration().V1alpha1().ValidatingAdmissionPolicies().Informer()),
		c.reconcilePolicyDefinition,
		generic.ControllerOptions{
			Workers: 1,
			Name:    "cel-policy-definitions",
		},
	)
	c.policyBindingController = generic.NewController(
		generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicyBinding](
			informerFactory.Admissionregistration().V1alpha1().ValidatingAdmissionPolicyBindings().Informer()),
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
			// TODO: add metrics for ignored error here
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
					action:  actionDeny,
					message: message,
				},
				definition: definition,
				binding:    binding,
			})
		default:
			deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
				policyDecision: policyDecision{
					action:  actionDeny,
					message: fmt.Errorf("unrecognized failure policy: '%v'", policy).Error(),
				},
				definition: definition,
				binding:    binding,
			})
		}
	}
	for definitionNamespacedName, definitionInfo := range c.definitionInfo {
		definition := definitionInfo.lastReconciledValue
		matches, matchKind, err := c.validatorCompiler.DefinitionMatches(a, o, definition)
		if err != nil {
			// Configuration error.
			addConfigError(err, definition, nil)
			continue
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
			matches, err := c.validatorCompiler.BindingMatches(a, o, binding)
			if err != nil {
				// Configuration error.
				addConfigError(err, definition, binding)
				continue
			}
			if !matches {
				continue
			}

			var param *unstructured.Unstructured

			// If definition has paramKind, paramRef is required in binding.
			// If definition has no paramKind, paramRef set in binding will be ignored.
			paramKind := definition.Spec.ParamKind
			paramRef := binding.Spec.ParamRef
			if paramKind != nil && paramRef != nil {

				// Find the params referred by the binding by looking its name up
				// in our informer for its CRD
				paramInfo, ok := c.paramsCRDControllers[*paramKind]
				if !ok {
					addConfigError(fmt.Errorf("paramKind kind `%v` not known",
						paramKind.String()), definition, binding)
					continue
				}

				// If the param informer for this admission policy has not yet
				// had time to perform an initial listing, don't attempt to use
				// it.
				//!TOOD(alexzielenski): add a wait for a very short amount of
				// time for the cache to sync
				if !paramInfo.controller.HasSynced() {
					addConfigError(fmt.Errorf("paramKind kind `%v` not yet synced to use for admission",
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

					if k8serrors.IsInvalid(err) {
						// Param mis-configured
						// require to set paramRef.namespace for namespaced resource and unset paramRef.namespace for cluster scoped resource
						continue
					} else if k8serrors.IsNotFound(err) {
						// Param not yet available. User may need to wait a bit
						// before being able to use it for validation.
						continue
					}

					// There was a bad internal error
					utilruntime.HandleError(err)
					continue
				}
			}

			validator := bindingInfo.validator.Load()
			if validator == nil {
				// Compile policy definition using binding
				newValidator := c.validatorCompiler.Compile(definition)
				validator = &newValidator

				bindingInfo.validator.Store(validator)
			}

			decisions, err := (*validator).Validate(a, o, param, matchKind)
			if err != nil {
				// runtime error. Apply failure policy
				wrappedError := fmt.Errorf("failed to evaluate CEL expression: %w", err)
				addConfigError(wrappedError, definition, binding)
				continue
			}

			for _, decision := range decisions {
				switch decision.action {
				case actionAdmit:
					if decision.evaluation == evalError {
						celmetrics.Metrics.ObserveAdmissionWithError(ctx, decision.elapsed, definition.Name, binding.Name, "active")
					}
				case actionDeny:
					deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
						definition:     definition,
						binding:        binding,
						policyDecision: decision,
					})
					celmetrics.Metrics.ObserveRejection(ctx, decision.elapsed, definition.Name, binding.Name, "active")
				default:
					return fmt.Errorf("unrecognized evaluation decision '%s' for ValidatingAdmissionPolicyBinding '%s' with ValidatingAdmissionPolicy '%s'",
						decision.action, binding.Name, definition.Name)
				}
			}
		}
	}

	if len(deniedDecisions) > 0 {
		// TODO: refactor admission.NewForbidden so the name extraction is reusable but the code/reason is customizable
		var message string
		deniedDecision := deniedDecisions[0]
		if deniedDecision.binding != nil {
			message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' with binding '%s' denied request: %s", deniedDecision.definition.Name, deniedDecision.binding.Name, deniedDecision.message)
		} else {
			message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' denied request: %s", deniedDecision.definition.Name, deniedDecision.message)
		}
		err := admission.NewForbidden(a, errors.New(message)).(*k8serrors.StatusError)
		reason := deniedDecision.reason
		if len(reason) == 0 {
			reason = metav1.StatusReasonInvalid
		}
		err.ErrStatus.Reason = reason
		err.ErrStatus.Code = reasonToCode(reason)
		err.ErrStatus.Details.Causes = append(err.ErrStatus.Details.Causes, metav1.StatusCause{Message: message})
		return err
	}
	return nil
}

func (c *celAdmissionController) HasSynced() bool {
	return c.policyBindingController.HasSynced() &&
		c.policyDefinitionsController.HasSynced()
}

func (c *celAdmissionController) ValidateInitialization() error {
	return c.validatorCompiler.ValidateInitialization()
}
