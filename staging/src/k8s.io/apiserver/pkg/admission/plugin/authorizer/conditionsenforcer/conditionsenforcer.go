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

package conditionsenforcer

import (
	"context"
	"errors"
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	apiscel "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

const (
	// PluginName indicates the name of admission plug-in
	PluginName = "AuthorizationConditionsEnforcer"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewConditionalAuthorizationEnforcer(false, true), nil // Feature default off, can be overridden by InspectFeatureGates, but CEL on by default
	})
}

// TODO(luxas): Add an integration test that it's not possible to intercept SAR or ACR using this admission controller
// TODO(luxas): Should VAP-related objects be added to the exclusion list for conditions? Most likely yes.

var _ admission.Interface = &conditionsEnforcer{}
var _ admission.ValidationInterface = &conditionsEnforcer{}
var _ genericadmissioninit.WantsFeatures = &conditionsEnforcer{}
var _ genericadmissioninit.WantsAuthorizer = &conditionsEnforcer{}
var _ genericadmissioninit.WantsExternalKubeInformerFactory = &conditionsEnforcer{}
var _ genericadmissioninit.WantsExternalKubeClientSet = &conditionsEnforcer{}

// NewConditionalAuthorizationEnforcer instantiates a new authorization conditions enforcer admission plugin
func NewConditionalAuthorizationEnforcer(featureEnabled, enableBuiltinCEL bool) *conditionsEnforcer {
	return &conditionsEnforcer{
		featureEnabled:   featureEnabled,
		enableBuiltinCEL: enableBuiltinCEL,
	}
}

type conditionsEnforcer struct {
	featureEnabled bool
	authorizer     authorizer.Authorizer

	enableBuiltinCEL bool
	informerFactory  informers.SharedInformerFactory
	client           kubernetes.Interface
}

func (c *conditionsEnforcer) InspectFeatureGates(features featuregate.FeatureGate) {
	c.featureEnabled = features.Enabled(genericfeatures.ConditionalAuthorization)
}

func (c *conditionsEnforcer) SetAuthorizer(authorizer authorizer.Authorizer) {
	c.authorizer = authorizer
}

func (c *conditionsEnforcer) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	c.informerFactory = f
}

func (c *conditionsEnforcer) SetExternalKubeClientSet(client kubernetes.Interface) {
	c.client = client
}

func (c *conditionsEnforcer) ValidateInitialization() error {
	if c.enableBuiltinCEL {
		if c.authorizer == nil {
			return errors.New("ConditionalAuthorizationEnforcer: authorizer must be given when enableBuiltinCEL == true")
		}
		if c.informerFactory == nil {
			return errors.New("ConditionalAuthorizationEnforcer: informerFactory must be given when enableBuiltinCEL == true")
		}
		if c.client == nil {
			return errors.New("ConditionalAuthorizationEnforcer: client must be given when enableBuiltinCEL == true")
		}
		// Register the CEL evaluator into the global registry.
		RegisterBuiltinConditionsEvaluator("cel", plugincel.NewAuthorizationConditionsEvaluator(c.authorizer, c.informerFactory, c.client, apiscel.RuntimeCELCostBudget), true)
	}
	return nil
}

func (c *conditionsEnforcer) Handles(operation admission.Operation) bool {
	return c.featureEnabled
}

func (c *conditionsEnforcer) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	authz, decisionToEnforce, ok := request.ConditionallyAuthorizedDecisionFrom(ctx)
	if !ok {
		// In the unconditionally authorized path, nothing is added to the context, hence this means "directly authorized"
		return nil
	}

	// The a.GetObject() and a.GetOldObject() objects are passed to admission using the internal API types. Before
	// calling EvaluateConditions(), we need to convert them to the request version. For a CRD, however, the
	// OldObject cannot be converted, and so it stays the same, or worse, partially broken.
	//
	// TODO(luxas): This should be using admission.NewVersionedAttributes, but that implementation for some reason
	// does not override GetOldObject, probably by error. That should probably be fixed, but I did not have the courage
	// right not to implement it, in case the bug would be load-bearing somewhere. Thus the versioned objects override
	// is done locally in this package with versionedAttributes, which copies the (short) implementation over 1:1.
	versionedAttributes, err := newVersionedAttributes(a, a.GetKind(), o)
	if err != nil {
		return fmt.Errorf("failed to convert objects to request version: %w", err)
	}

	data := authorizer.ConditionsData{
		AdmissionControl: versionedAttributes,
	}

	decision, reason, err := c.evaluateConditions(ctx, authz, decisionToEnforce, data)

	// The code flow here should exactly match filters.WithAuthorization.
	// an authorizer could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
	if decision == authorizer.DecisionAllow {
		audit.AddAuditAnnotations(ctx,
			filters.DecisionAnnotationKey, filters.DecisionAllow,
			filters.ReasonAnnotationKey, reason)
		return nil
	}

	if err != nil {
		audit.AddAuditAnnotation(ctx, filters.ReasonAnnotationKey, filters.ReasonError)
		return apierrors.NewInternalError(err)
	}

	authzAttrs, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get authorizer attributes: %w", err)
	}

	klog.V(4).InfoS("Forbidden (during conditional authorization)", "URI", authzAttrs.GetPath(), "reason", reason)
	audit.AddAuditAnnotations(ctx,
		filters.DecisionAnnotationKey, filters.DecisionForbid,
		filters.ReasonAnnotationKey, reason)

	return apierrors.NewForbidden(versionedAttributes.GetResource().GroupResource(), versionedAttributes.GetName(), responsewriters.ForbiddenStatusError(authzAttrs, reason))
}

func (c *conditionsEnforcer) evaluateConditions(ctx context.Context, authorizer authorizer.Authorizer, unevaluatedDecision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	// This should not be the case, but if somehow an Allow/Deny/NoOpinion decision shows up here, there's nothing to evaluate.
	if unevaluatedDecision.IsUnconditional() {
		return unevaluatedDecision.UnconditionalParts()
	}

	return authorizer.EvaluateConditions(ctx, unevaluatedDecision, data)
}

// newVersionedAttributes returns versioned attributes with the old and new object (if non-nil) converted to the requested kind
func newVersionedAttributes(attr admission.Attributes, gvk schema.GroupVersionKind, o admission.ObjectInterfaces) (*versionedAttributes, error) {
	// convert the old and new objects to the requested version
	versionedAttr := &versionedAttributes{
		Attributes: attr,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := admission.ConvertToGVK(oldObj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedOldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := admission.ConvertToGVK(obj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedObject = out
	}
	return versionedAttr, nil
}

// versionedAttributes is a wrapper around the original admission attributes, adding versioned
// variants of the object and old object.
type versionedAttributes struct {
	// Attributes holds the original admission attributes
	admission.Attributes
	// VersionedOldObject holds Attributes.OldObject (if non-nil), converted to VersionedKind.
	// It must never be mutated.
	VersionedOldObject runtime.Object
	// VersionedObject holds Attributes.Object (if non-nil), converted to VersionedKind.
	// If mutated, Dirty must be set to true by the mutator.
	VersionedObject runtime.Object
}

// GetObject overrides the Attributes.GetObject()
func (v *versionedAttributes) GetObject() runtime.Object {
	if v.VersionedObject != nil {
		return v.VersionedObject
	}
	return v.Attributes.GetObject()
}

// GetOldObject overrides the Attributes.GetOldObject()
func (v *versionedAttributes) GetOldObject() runtime.Object {
	if v.VersionedOldObject != nil {
		return v.VersionedOldObject
	}
	return v.Attributes.GetOldObject()
}
