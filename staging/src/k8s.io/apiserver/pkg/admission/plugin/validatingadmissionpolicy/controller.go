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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/api/admissionregistration/v1beta1"
	v1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	celmetrics "k8s.io/apiserver/pkg/admission/cel"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/internal/generic"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/matching"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
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

	authz authorizer.Authorizer
}

// Everything someone might need to validate a single ValidatingPolicyDefinition
// against all of its registered bindings.
type policyData struct {
	definitionInfo
	paramInfo
	bindings []bindingInfo
}

// contains the cel PolicyDecisions along with the ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding
// that determined the decision
type policyDecisionWithMetadata struct {
	PolicyDecision
	Definition *v1beta1.ValidatingAdmissionPolicy
	Binding    *v1beta1.ValidatingAdmissionPolicyBinding
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
	lastReconciledValue *v1beta1.ValidatingAdmissionPolicy
}

type bindingInfo struct {
	// Compiled CEL expression turned into an validator
	validator Validator

	// Last value seen by this controller to be used in policy enforcement
	// May not be nil
	lastReconciledValue *v1beta1.ValidatingAdmissionPolicyBinding
}

type paramInfo struct {
	// Controller which is watching this param CRD
	controller generic.Controller[runtime.Object]

	// Function to call to stop the informer and clean up the controller
	stop func()

	// Whether this param is cluster or namespace scoped
	scope meta.RESTScope

	// Policy Definitions which refer to this param CRD
	dependentDefinitions sets.Set[namespacedName]
}

func NewAdmissionController(
	// Injected Dependencies
	informerFactory informers.SharedInformerFactory,
	client kubernetes.Interface,
	restMapper meta.RESTMapper,
	dynamicClient dynamic.Interface,
	authz authorizer.Authorizer,
) CELPolicyEvaluator {
	return &celAdmissionController{
		definitions: atomic.Value{},
		policyController: newPolicyController(
			restMapper,
			client,
			dynamicClient,
			informerFactory,
			nil,
			NewMatcher(matching.NewMatcher(informerFactory.Core().V1().Namespaces().Lister(), client)),
			generic.NewInformer[*v1beta1.ValidatingAdmissionPolicy](
				informerFactory.Admissionregistration().V1beta1().ValidatingAdmissionPolicies().Informer()),
			generic.NewInformer[*v1beta1.ValidatingAdmissionPolicyBinding](
				informerFactory.Admissionregistration().V1beta1().ValidatingAdmissionPolicyBindings().Informer()),
		),
		authz: authz,
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

const maxAuditAnnotationValueLength = 10 * 1024

func (c *celAdmissionController) Validate(
	ctx context.Context,
	a admission.Attributes,
	o admission.ObjectInterfaces,
) (err error) {
	if !c.HasSynced() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	var deniedDecisions []policyDecisionWithMetadata

	addConfigError := func(err error, definition *v1beta1.ValidatingAdmissionPolicy, binding *v1beta1.ValidatingAdmissionPolicyBinding) {
		// we always default the FailurePolicy if it is unset and validate it in API level
		var policy v1beta1.FailurePolicyType
		if definition.Spec.FailurePolicy == nil {
			policy = v1beta1.Fail
		} else {
			policy = *definition.Spec.FailurePolicy
		}

		// apply FailurePolicy specified in ValidatingAdmissionPolicy, the default would be Fail
		switch policy {
		case v1beta1.Ignore:
			// TODO: add metrics for ignored error here
			return
		case v1beta1.Fail:
			var message string
			if binding == nil {
				message = fmt.Errorf("failed to configure policy: %w", err).Error()
			} else {
				message = fmt.Errorf("failed to configure binding: %w", err).Error()
			}
			deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
				PolicyDecision: PolicyDecision{
					Action:  ActionDeny,
					Message: message,
				},
				Definition: definition,
				Binding:    binding,
			})
		default:
			deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
				PolicyDecision: PolicyDecision{
					Action:  ActionDeny,
					Message: fmt.Errorf("unrecognized failure policy: '%v'", policy).Error(),
				},
				Definition: definition,
				Binding:    binding,
			})
		}
	}
	policyDatas := c.definitions.Load().([]policyData)

	authz := newCachingAuthorizer(c.authz)

	for _, definitionInfo := range policyDatas {
		// versionedAttributes will be set to non-nil inside of the loop, but
		// is scoped outside of the param loop so we only convert once. We defer
		// conversion so that it is only performed when we know a policy matches,
		// saving the cost of converting non-matching requests.
		var versionedAttr *admission.VersionedAttributes

		definition := definitionInfo.lastReconciledValue
		matches, matchResource, matchKind, err := c.policyController.matcher.DefinitionMatches(a, o, definition)
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

		auditAnnotationCollector := newAuditAnnotationCollector()
		for _, bindingInfo := range definitionInfo.bindings {
			// If the key is inside dependentBindings, there is guaranteed to
			// be a bindingInfo for it
			binding := bindingInfo.lastReconciledValue
			matches, err := c.policyController.matcher.BindingMatches(a, o, binding)
			if err != nil {
				// Configuration error.
				addConfigError(err, definition, binding)
				continue
			}
			if !matches {
				continue
			}

			params, err := c.collectParams(definition.Spec.ParamKind, definitionInfo.paramInfo, binding.Spec.ParamRef, a.GetNamespace())
			if err != nil {
				addConfigError(err, definition, binding)
				continue
			} else if versionedAttr == nil && len(params) > 0 {
				// As optimization versionedAttr creation is deferred until
				// first use. Since > 0 params, we will validate
				va, err := admission.NewVersionedAttributes(a, matchKind, o)
				if err != nil {
					wrappedErr := fmt.Errorf("failed to convert object version: %w", err)
					addConfigError(wrappedErr, definition, binding)
					continue
				}
				versionedAttr = va
			}

			var validationResults []ValidateResult
			var namespace *v1.Namespace
			namespaceName := a.GetNamespace()

			// Special case, the namespace object has the namespace of itself (maybe a bug).
			// unset it if the incoming object is a namespace
			if gvk := a.GetKind(); gvk.Kind == "Namespace" && gvk.Version == "v1" && gvk.Group == "" {
				namespaceName = ""
			}

			// if it is cluster scoped, namespaceName will be empty
			// Otherwise, get the Namespace resource.
			if namespaceName != "" {
				namespace, err = c.policyController.matcher.GetNamespace(namespaceName)
				if err != nil {
					return err
				}
			}

			for _, param := range params {
				var p runtime.Object = param
				if p != nil && p.GetObjectKind().GroupVersionKind().Empty() {
					// Make sure param has TypeMeta populated
					// This is a simple hack to make sure typeMeta is
					// available to CEL without making copies of objects, etc.
					p = &wrappedParam{
						TypeMeta: metav1.TypeMeta{
							APIVersion: definition.Spec.ParamKind.APIVersion,
							Kind:       definition.Spec.ParamKind.Kind,
						},
						nested: param,
					}
				}
				validationResults = append(validationResults, bindingInfo.validator.Validate(ctx, matchResource, versionedAttr, p, namespace, celconfig.RuntimeCELCostBudget, authz))
			}

			for _, validationResult := range validationResults {
				for i, decision := range validationResult.Decisions {
					switch decision.Action {
					case ActionAdmit:
						if decision.Evaluation == EvalError {
							celmetrics.Metrics.ObserveAdmissionWithError(ctx, decision.Elapsed, definition.Name, binding.Name, "active")
						}
					case ActionDeny:
						for _, action := range binding.Spec.ValidationActions {
							switch action {
							case v1beta1.Deny:
								deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
									Definition:     definition,
									Binding:        binding,
									PolicyDecision: decision,
								})
								celmetrics.Metrics.ObserveRejection(ctx, decision.Elapsed, definition.Name, binding.Name, "active")
							case v1beta1.Audit:
								c.publishValidationFailureAnnotation(binding, i, decision, versionedAttr)
								celmetrics.Metrics.ObserveAudit(ctx, decision.Elapsed, definition.Name, binding.Name, "active")
							case v1beta1.Warn:
								warning.AddWarning(ctx, "", fmt.Sprintf("Validation failed for ValidatingAdmissionPolicy '%s' with binding '%s': %s", definition.Name, binding.Name, decision.Message))
								celmetrics.Metrics.ObserveWarn(ctx, decision.Elapsed, definition.Name, binding.Name, "active")
							}
						}
					default:
						return fmt.Errorf("unrecognized evaluation decision '%s' for ValidatingAdmissionPolicyBinding '%s' with ValidatingAdmissionPolicy '%s'",
							decision.Action, binding.Name, definition.Name)
					}
				}

				for _, auditAnnotation := range validationResult.AuditAnnotations {
					switch auditAnnotation.Action {
					case AuditAnnotationActionPublish:
						value := auditAnnotation.Value
						if len(auditAnnotation.Value) > maxAuditAnnotationValueLength {
							value = value[:maxAuditAnnotationValueLength]
						}
						auditAnnotationCollector.add(auditAnnotation.Key, value)
					case AuditAnnotationActionError:
						// When failurePolicy=fail, audit annotation errors result in deny
						deniedDecisions = append(deniedDecisions, policyDecisionWithMetadata{
							Definition: definition,
							Binding:    binding,
							PolicyDecision: PolicyDecision{
								Action:     ActionDeny,
								Evaluation: EvalError,
								Message:    auditAnnotation.Error,
								Elapsed:    auditAnnotation.Elapsed,
							},
						})
						celmetrics.Metrics.ObserveRejection(ctx, auditAnnotation.Elapsed, definition.Name, binding.Name, "active")
					case AuditAnnotationActionExclude: // skip it
					default:
						return fmt.Errorf("unsupported AuditAnnotation Action: %s", auditAnnotation.Action)
					}
				}
			}
		}
		auditAnnotationCollector.publish(definition.Name, a)
	}

	if len(deniedDecisions) > 0 {
		// TODO: refactor admission.NewForbidden so the name extraction is reusable but the code/reason is customizable
		var message string
		deniedDecision := deniedDecisions[0]
		if deniedDecision.Binding != nil {
			message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' with binding '%s' denied request: %s", deniedDecision.Definition.Name, deniedDecision.Binding.Name, deniedDecision.Message)
		} else {
			message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' denied request: %s", deniedDecision.Definition.Name, deniedDecision.Message)
		}
		err := admission.NewForbidden(a, errors.New(message)).(*k8serrors.StatusError)
		reason := deniedDecision.Reason
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

// Returns objects to use to evaluate the policy
func (c *celAdmissionController) collectParams(
	paramKind *v1beta1.ParamKind,
	info paramInfo,
	paramRef *v1beta1.ParamRef,
	namespace string,
) ([]runtime.Object, error) {
	// If definition has paramKind, paramRef is required in binding.
	// If definition has no paramKind, paramRef set in binding will be ignored.
	var params []runtime.Object
	var paramStore generic.NamespacedLister[runtime.Object]

	// Make sure the param kind is ready to use
	if paramKind != nil && paramRef != nil {
		if info.controller == nil {
			return nil, fmt.Errorf("paramKind kind `%v` not known",
				paramKind.String())
		}

		// Set up cluster-scoped, or namespaced access to the params
		// "default" if not provided, and paramKind is namespaced
		paramStore = info.controller.Informer()
		if info.scope.Name() == meta.RESTScopeNameNamespace {
			paramsNamespace := namespace
			if len(paramRef.Namespace) > 0 {
				paramsNamespace = paramRef.Namespace
			} else if len(paramsNamespace) == 0 {
				// You must supply namespace if your matcher can possibly
				// match a cluster-scoped resource
				return nil, fmt.Errorf("cannot use namespaced paramRef in policy binding that matches cluster-scoped resources")
			}

			paramStore = info.controller.Informer().Namespaced(paramsNamespace)
		}

		// If the param informer for this admission policy has not yet
		// had time to perform an initial listing, don't attempt to use
		// it.
		timeoutCtx, cancel := context.WithTimeout(c.policyController.context, 1*time.Second)
		defer cancel()

		if !cache.WaitForCacheSync(timeoutCtx.Done(), info.controller.HasSynced) {
			return nil, fmt.Errorf("paramKind kind `%v` not yet synced to use for admission",
				paramKind.String())
		}
	}

	// Find params to use with policy
	switch {
	case paramKind == nil:
		// ParamKind is unset. Ignore any globalParamRef or namespaceParamRef
		// setting.
		return []runtime.Object{nil}, nil
	case paramRef == nil:
		// Policy ParamKind is set, but binding does not use it.
		// Validate with nil params
		return []runtime.Object{nil}, nil
	case len(paramRef.Namespace) > 0 && info.scope.Name() == meta.RESTScopeRoot.Name():
		// Not allowed to set namespace for cluster-scoped param
		return nil, fmt.Errorf("paramRef.namespace must not be provided for a cluster-scoped `paramKind`")

	case len(paramRef.Name) > 0:
		if paramRef.Selector != nil {
			// This should be validated, but just in case.
			return nil, fmt.Errorf("paramRef.name and paramRef.selector are mutually exclusive")
		}

		switch param, err := paramStore.Get(paramRef.Name); {
		case err == nil:
			params = []runtime.Object{param}
		case k8serrors.IsNotFound(err):
			// Param not yet available. User may need to wait a bit
			// before being able to use it for validation.
			//
			// Set params to nil to prepare for not found action
			params = nil
		case k8serrors.IsInvalid(err):
			// Param mis-configured
			// require to set namespace for namespaced resource
			// and unset namespace for cluster scoped resource
			return nil, err
		default:
			// Internal error
			utilruntime.HandleError(err)
			return nil, err
		}
	case paramRef.Selector != nil:
		// Select everything by default if empty name and selector
		selector, err := metav1.LabelSelectorAsSelector(paramRef.Selector)
		if err != nil {
			// Cannot parse label selector: configuration error
			return nil, err

		}

		paramList, err := paramStore.List(selector)
		if err != nil {
			// There was a bad internal error
			utilruntime.HandleError(err)
			return nil, err
		}

		// Successfully grabbed params
		params = paramList
	default:
		// Should be unreachable due to validation
		return nil, fmt.Errorf("one of name or selector must be provided")
	}

	// Apply fail action for params not found case
	if len(params) == 0 && paramRef.ParameterNotFoundAction != nil && *paramRef.ParameterNotFoundAction == v1beta1.DenyAction {
		return nil, errors.New("no params found for policy binding with `Deny` parameterNotFoundAction")
	}

	return params, nil
}

func (c *celAdmissionController) publishValidationFailureAnnotation(binding *v1beta1.ValidatingAdmissionPolicyBinding, expressionIndex int, decision PolicyDecision, attributes admission.Attributes) {
	key := "validation.policy.admission.k8s.io/validation_failure"
	// Marshal to a list of failures since, in the future, we may need to support multiple failures
	valueJson, err := utiljson.Marshal([]validationFailureValue{{
		ExpressionIndex:   expressionIndex,
		Message:           decision.Message,
		ValidationActions: binding.Spec.ValidationActions,
		Binding:           binding.Name,
		Policy:            binding.Spec.PolicyName,
	}})
	if err != nil {
		klog.Warningf("Failed to set admission audit annotation %s for ValidatingAdmissionPolicy %s and ValidatingAdmissionPolicyBinding %s: %v", key, binding.Spec.PolicyName, binding.Name, err)
	}
	value := string(valueJson)
	if err := attributes.AddAnnotation(key, value); err != nil {
		klog.Warningf("Failed to set admission audit annotation %s to %s for ValidatingAdmissionPolicy %s and ValidatingAdmissionPolicyBinding %s: %v", key, value, binding.Spec.PolicyName, binding.Name, err)
	}
}

func (c *celAdmissionController) HasSynced() bool {
	return c.policyController.HasSynced() && c.definitions.Load() != nil
}

func (c *celAdmissionController) ValidateInitialization() error {
	return c.policyController.matcher.ValidateInitialization()
}

func (c *celAdmissionController) refreshPolicies() {
	c.definitions.Store(c.policyController.latestPolicyData())
}

// validationFailureValue defines the JSON format of a "validation.policy.admission.k8s.io/validation_failure" audit
// annotation value.
type validationFailureValue struct {
	Message           string                     `json:"message"`
	Policy            string                     `json:"policy"`
	Binding           string                     `json:"binding"`
	ExpressionIndex   int                        `json:"expressionIndex"`
	ValidationActions []v1beta1.ValidationAction `json:"validationActions"`
}

type auditAnnotationCollector struct {
	annotations map[string][]string
}

func newAuditAnnotationCollector() auditAnnotationCollector {
	return auditAnnotationCollector{annotations: map[string][]string{}}
}

func (a auditAnnotationCollector) add(key, value string) {
	// If multiple bindings produces the exact same key and value for an audit annotation,
	// ignore the duplicates.
	for _, v := range a.annotations[key] {
		if v == value {
			return
		}
	}
	a.annotations[key] = append(a.annotations[key], value)
}

func (a auditAnnotationCollector) publish(policyName string, attributes admission.Attributes) {
	for key, bindingAnnotations := range a.annotations {
		var value string
		if len(bindingAnnotations) == 1 {
			value = bindingAnnotations[0]
		} else {
			// Multiple distinct values can exist when binding params are used in the valueExpression of an auditAnnotation.
			// When this happens, the values are concatenated into a comma-separated list.
			value = strings.Join(bindingAnnotations, ", ")
		}
		if err := attributes.AddAnnotation(policyName+"/"+key, value); err != nil {
			klog.Warningf("Failed to set admission audit annotation %s to %s for ValidatingAdmissionPolicy %s: %v", key, value, policyName, err)
		}
	}
}

// A workaround to fact that native types do not have TypeMeta populated, which
// is needed for CEL expressions to be able to access the value.
type wrappedParam struct {
	metav1.TypeMeta
	nested runtime.Object
}

func (w *wrappedParam) MarshalJSON() ([]byte, error) {
	return nil, errors.New("MarshalJSON unimplemented for wrappedParam")
}

func (w *wrappedParam) UnmarshalJSON(data []byte) error {
	return errors.New("UnmarshalJSON unimplemented for wrappedParam")
}

func (w *wrappedParam) ToUnstructured() interface{} {
	res, err := runtime.DefaultUnstructuredConverter.ToUnstructured(w.nested)

	if err != nil {
		return nil
	}

	metaRes, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&w.TypeMeta)
	if err != nil {
		return nil
	}

	for k, v := range metaRes {
		res[k] = v
	}

	return res
}

func (w *wrappedParam) DeepCopyObject() runtime.Object {
	return &wrappedParam{
		TypeMeta: w.TypeMeta,
		nested:   w.nested.DeepCopyObject(),
	}
}

func (w *wrappedParam) GetObjectKind() schema.ObjectKind {
	return w
}
