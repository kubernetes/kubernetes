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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/matching"

	"k8s.io/api/admissionregistration/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

var _ CELPolicyEvaluator = &celAdmissionController{}

// celAdmissionController is the top-level controller for admission control using CEL
// it is responsible for watching policy definitions, bindings, and config param CRDs
type celAdmissionController struct {
	// Controller which manages book-keeping for the cluster's dynamic policy
	// information.
	policyController *policyController

	// atomic []policyData
	// list of every known policy definition, and all informatoin required to
	// validate its bindings against an object.
	// A snapshot of the current policy configuration is synced with this field
	// asynchronously
	definitions atomic.Value
}

// Everything someone might need to validate a single ValidatingPolicyDefinition
// against all of its registered bindings.
type policyData struct {
	definitionInfo
	paramController generic.Controller[*unstructured.Unstructured]
	bindings        []bindingInfo
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
	dependentDefinitions sets.Set[namespacedName]
}

func NewAdmissionController(
	// Injected Dependencies
	informerFactory informers.SharedInformerFactory,
	client kubernetes.Interface,
	restMapper meta.RESTMapper,
	dynamicClient dynamic.Interface,
) CELPolicyEvaluator {
	return &celAdmissionController{
		definitions: atomic.Value{},
		policyController: newPolicyController(
			restMapper,
			dynamicClient,
			&CELValidatorCompiler{
				Matcher: matching.NewMatcher(informerFactory.Core().V1().Namespaces().Lister(), client),
			},
			generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicy](
				informerFactory.Admissionregistration().V1alpha1().ValidatingAdmissionPolicies().Informer()),
			generic.NewInformer[*v1alpha1.ValidatingAdmissionPolicyBinding](
				informerFactory.Admissionregistration().V1alpha1().ValidatingAdmissionPolicyBindings().Informer()),
		),
	}
}

func (c *celAdmissionController) Run(stopCh <-chan struct{}) {
	ctx, cancel := context.WithCancel(context.Background())
	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.policyController.Run(ctx)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()

		// Wait indefinitely until policies/bindings are listed & handled before
		// allowing policies to be refreshed
		if !cache.WaitForNamedCacheSync("cel-admission-controller", ctx.Done(), c.policyController.HasSynced) {
			return
		}

		// Loop every 1 second until context is cancelled, refreshing policies
		wait.Until(c.refreshPolicies, 1*time.Second, ctx.Done())
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
	if !c.HasSynced() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

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
	policyDatas := c.definitions.Load().([]policyData)

	for _, definitionInfo := range policyDatas {
		definition := definitionInfo.lastReconciledValue
		matches, matchKind, err := c.policyController.DefinitionMatches(a, o, definition)
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

		for _, bindingInfo := range definitionInfo.bindings {
			// If the key is inside dependentBindings, there is guaranteed to
			// be a bindingInfo for it
			binding := bindingInfo.lastReconciledValue
			matches, err := c.policyController.BindingMatches(a, o, binding)
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
				paramController := definitionInfo.paramController
				if paramController == nil {
					addConfigError(fmt.Errorf("paramKind kind `%v` not known",
						paramKind.String()), definition, binding)
					continue
				}

				// If the param informer for this admission policy has not yet
				// had time to perform an initial listing, don't attempt to use
				// it.
				timeoutCtx, cancel := context.WithTimeout(c.policyController.context, 1*time.Second)
				defer cancel()

				if !cache.WaitForCacheSync(timeoutCtx.Done(), paramController.HasSynced) {
					addConfigError(fmt.Errorf("paramKind kind `%v` not yet synced to use for admission",
						paramKind.String()), definition, binding)
					continue
				}

				if len(paramRef.Namespace) == 0 {
					param, err = paramController.Informer().Get(paramRef.Name)
				} else {
					param, err = paramController.Informer().Namespaced(paramRef.Namespace).Get(paramRef.Name)
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
			decisions, err := bindingInfo.validator.Validate(a, o, param, matchKind)
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
	return c.policyController.HasSynced() && c.definitions.Load() != nil
}

func (c *celAdmissionController) ValidateInitialization() error {
	return c.policyController.ValidateInitialization()
}

func (c *celAdmissionController) refreshPolicies() {
	c.definitions.Store(c.policyController.latestPolicyData())
}
