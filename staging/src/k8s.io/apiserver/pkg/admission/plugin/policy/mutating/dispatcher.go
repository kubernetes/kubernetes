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

package mutating

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/api/admissionregistration/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	webhookgeneric "k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func NewDispatcher(a authorizer.Authorizer, m *matching.Matcher, tcm patch.TypeConverterManager) generic.Dispatcher[PolicyHook] {
	res := &dispatcher{
		matcher: m,
		//!TODO: pass in static type converter to reduce network calls
		typeConverterManager: tcm,
	}
	res.Dispatcher = generic.NewPolicyDispatcher[*Policy, *PolicyBinding, PolicyEvaluator](
		NewMutatingAdmissionPolicyAccessor,
		NewMutatingAdmissionPolicyBindingAccessor,
		m,
		res.dispatchInvocations,
	)
	return res
}

type dispatcher struct {
	matcher              *matching.Matcher
	typeConverterManager patch.TypeConverterManager
	generic.Dispatcher[PolicyHook]
}

func (d *dispatcher) Run(ctx context.Context) error {
	go d.typeConverterManager.Run(ctx)
	return d.Dispatcher.Run(ctx)
}

func (d *dispatcher) dispatchInvocations(
	ctx context.Context,
	a admission.Attributes,
	o admission.ObjectInterfaces,
	versionedAttributes webhookgeneric.VersionedAttributeAccessor,
	invocations []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator],
) ([]generic.PolicyError, *k8serrors.StatusError) {
	var lastVersionedAttr *admission.VersionedAttributes

	reinvokeCtx := a.GetReinvocationContext()
	var policyReinvokeCtx *policyReinvokeContext
	if v := reinvokeCtx.Value(PluginName); v != nil {
		policyReinvokeCtx = v.(*policyReinvokeContext)
	} else {
		policyReinvokeCtx = &policyReinvokeContext{}
		reinvokeCtx.SetValue(PluginName, policyReinvokeCtx)
	}

	if reinvokeCtx.IsReinvoke() && policyReinvokeCtx.IsOutputChangedSinceLastPolicyInvocation(a.GetObject()) {
		// If the object has changed, we know the in-tree plugin re-invocations have mutated the object,
		// and we need to reinvoke all eligible policies.
		policyReinvokeCtx.RequireReinvokingPreviouslyInvokedPlugins()
	}
	defer func() {
		policyReinvokeCtx.SetLastPolicyInvocationOutput(a.GetObject())
	}()

	var policyErrors []generic.PolicyError
	addConfigError := func(err error, invocation generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator], reason metav1.StatusReason) {
		policyErrors = append(policyErrors, generic.PolicyError{
			Message: err,
			Policy:  NewMutatingAdmissionPolicyAccessor(invocation.Policy),
			Binding: NewMutatingAdmissionPolicyBindingAccessor(invocation.Binding),
			Reason:  reason,
		})
	}

	// There is at least one invocation to invoke. Make sure we have a namespace
	// object if the incoming object is not cluster scoped to pass into the evaluator.
	var namespace *v1.Namespace
	var err error
	namespaceName := a.GetNamespace()

	// Special case, the namespace object has the namespace of itself (maybe a bug).
	// unset it if the incoming object is a namespace
	if gvk := a.GetKind(); gvk.Kind == "Namespace" && gvk.Version == "v1" && gvk.Group == "" {
		namespaceName = ""
	}

	// if it is cluster scoped, namespaceName will be empty
	// Otherwise, get the Namespace resource.
	if namespaceName != "" {
		namespace, err = d.matcher.GetNamespace(namespaceName)
		if err != nil {
			return nil, k8serrors.NewNotFound(schema.GroupResource{Group: "", Resource: "namespaces"}, namespaceName)
		}
	}

	// Should loop through invocations, handling possible error and invoking
	// evaluator to apply patch, also should handle re-invocations
	for _, invocation := range invocations {
		if len(invocation.Evaluator) != len(invocation.Policy.Spec.Mutations) {
			// This would be a bug. The compiler should always return exactly as
			// many evaluators as there are mutations
			return nil, k8serrors.NewInternalError(fmt.Errorf("expected %v compiled evaluators for policy %v, got %v",
				invocation.Policy.Name, len(invocation.Policy.Spec.Mutations), len(invocation.Evaluator)))
		}

		for mutationIndex, mutation := range invocation.Policy.Spec.Mutations {
			invocationKey, err := keyFor(invocation, mutationIndex)
			if err != nil {
				// This should never happen. It occurs if there is a programming
				// error causing the Param not to be a valid object.
				return nil, k8serrors.NewInternalError(err)
			}

			if reinvokeCtx.IsReinvoke() && !policyReinvokeCtx.ShouldReinvoke(invocationKey) {
				continue
			}

			versionedAttr, err := versionedAttributes.VersionedAttribute(invocation.Kind)
			if err != nil {
				// This should never happen, we pre-warm versoined attribute
				// accessors before starting the dispatcher
				return nil, k8serrors.NewInternalError(err)
			}
			lastVersionedAttr = versionedAttr

			changed, err := d.dispatchOne(ctx, a, o, versionedAttr, namespace, invocation.Resource, invocation.Param, invocation.Evaluator[mutationIndex])
			if err != nil {
				var statusError *k8serrors.StatusError
				if errors.As(err, &statusError) {
					return nil, statusError
				}

				addConfigError(err, invocation, metav1.StatusReasonInvalid)
				continue
			}

			if changed {
				// Patch had changed the object. Prepare to reinvoke all previous webhooks that are eligible for re-invocation.
				policyReinvokeCtx.RequireReinvokingPreviouslyInvokedPlugins()
				reinvokeCtx.SetShouldReinvoke()
			}
			if mutation.ReinvocationPolicy != nil && *mutation.ReinvocationPolicy == v1alpha1.IfNeededReinvocationPolicy {
				policyReinvokeCtx.AddReinvocablePolicyToPreviouslyInvoked(invocationKey)
			}
		}
	}

	if lastVersionedAttr != nil && lastVersionedAttr.VersionedObject != nil && lastVersionedAttr.Dirty {
		if err := o.GetObjectConvertor().Convert(lastVersionedAttr.VersionedObject, lastVersionedAttr.Attributes.GetObject(), nil); err != nil {
			return nil, k8serrors.NewInternalError(fmt.Errorf("failed to convert object: %w", err))
		}
	}

	return policyErrors, nil
}

func (d *dispatcher) dispatchOne(
	ctx context.Context,
	a admission.Attributes,
	o admission.ObjectInterfaces,
	versionedAttributes *admission.VersionedAttributes,
	namespace *v1.Namespace,
	resource schema.GroupVersionResource,
	param runtime.Object,
	evaluator MutationEvaluationFunc,
) (changed bool, err error) {
	if evaluator == nil {
		// internal error. this should not happen
		return false, k8serrors.NewInternalError(fmt.Errorf("policy evaluator is nil"))
	}

	// Find type converter for the invoked Group-Version.
	typeConverter := d.typeConverterManager.GetTypeConverter(versionedAttributes.VersionedKind)
	if typeConverter == nil {
		// This can happen if the request is for a resource whose schema
		// has not been registered with the type converter manager.
		return false, k8serrors.NewServiceUnavailable(fmt.Sprintf("failed to get type converter for %s", versionedAttributes.VersionedKind))
	}

	newVersionedObject, err := evaluator(
		ctx,
		resource,
		versionedAttributes,
		o,
		param,
		namespace,
		typeConverter,
		celconfig.RuntimeCELCostBudget,
	)
	if err != nil {
		return false, err
	}

	changed = !apiequality.Semantic.DeepEqual(versionedAttributes.VersionedObject, newVersionedObject)
	versionedAttributes.Dirty = true
	versionedAttributes.VersionedObject = newVersionedObject
	o.GetObjectDefaulter().Default(newVersionedObject)
	return changed, nil
}

func keyFor(invocation generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator], mutatingIndex int) (key, error) {
	var paramUID types.NamespacedName
	if invocation.Param != nil {
		paramAccessor, err := meta.Accessor(invocation.Param)
		if err != nil {
			// This should never happen, as the param should have been validated
			// before being passed to the plugin.
			return key{}, err
		}
		paramUID = types.NamespacedName{
			Name:      paramAccessor.GetName(),
			Namespace: paramAccessor.GetNamespace(),
		}
	}

	return key{
		PolicyUID: types.NamespacedName{
			Name:      invocation.Policy.GetName(),
			Namespace: invocation.Policy.GetNamespace(),
		},
		BindingUID: types.NamespacedName{
			Name:      invocation.Binding.GetName(),
			Namespace: invocation.Binding.GetNamespace(),
		},
		ParamUID:      paramUID,
		MutationIndex: mutatingIndex,
	}, nil
}
