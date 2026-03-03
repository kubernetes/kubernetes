package conditionsenforcer

import (
	"context"
	"errors"
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
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
		return NewConditionalAuthorizationEnforcer(true), nil
	})
}

// TODO: Should we opt out of enforcing conditions for authorization-related resources?
// Or actually, it could be useful as well, to say that someone can only request some specific SAR requests.
/*var optOutGVRs = sets.New(
	schema.GroupVersionResource{Group: authorizationv1.SchemeGroupVersion.Group, Version: authorizationv1.SchemeGroupVersion.Version, Resource: "selfsubjectaccessreviews"},
)*/

// TODO: Add an integration test that it's not possible to intercept SAR or ACR using this admission controller

var _ admission.Interface = &lazilyInitializedPlugin{}
var _ admission.ValidationInterface = &lazilyInitializedPlugin{}

var _ genericadmissioninit.WantsFeatures = &lazilyInitializedPlugin{}
var _ genericadmissioninit.WantsAuthorizer = &lazilyInitializedPlugin{}
var _ genericadmissioninit.WantsExternalKubeInformerFactory = &lazilyInitializedPlugin{}
var _ genericadmissioninit.WantsExternalKubeClientSet = &lazilyInitializedPlugin{}

func NewConditionalAuthorizationEnforcer(enableBuiltinCEL bool) *lazilyInitializedPlugin {
	return &lazilyInitializedPlugin{
		enableBuiltinCEL: enableBuiltinCEL,
		featureEnabled:   false, // Default, can be overridden by InspectFeatureGates
	}
}

// TODO: Add VAP-related resources the exclusion list too?

type lazilyInitializedPlugin struct {
	builtinConditionSetEvaluators []authorizer.BuiltinConditionSetEvaluator
	featureEnabled                bool
	authorizer                    authorizer.Authorizer
	enableBuiltinCEL              bool
	informerFactory               informers.SharedInformerFactory
	client                        kubernetes.Interface
}

func (c *lazilyInitializedPlugin) InspectFeatureGates(features featuregate.FeatureGate) {
	c.featureEnabled = features.Enabled(genericfeatures.ConditionalAuthorization)
}

func (c *lazilyInitializedPlugin) SetAuthorizer(authorizer authorizer.Authorizer) {
	c.authorizer = authorizer
}

func (c *lazilyInitializedPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	c.informerFactory = f
}

func (c *lazilyInitializedPlugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	c.client = client
}

func (c *lazilyInitializedPlugin) ValidateInitialization() error {
	// Note: Whenever a new builtin evaluator is added here, it MUST also be added to withAuthorization in
	// staging/src/k8s.io/apiserver/pkg/endpoints/handlers/update.go
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

		c.builtinConditionSetEvaluators = append(c.builtinConditionSetEvaluators, NewCELBuiltinConditionSetEvaluator(c.authorizer, c.informerFactory, c.client))
	}

	return nil
}

func (c *lazilyInitializedPlugin) Handles(operation admission.Operation) bool {
	return c.featureEnabled
}

func (c *lazilyInitializedPlugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	authorizer, unevaluatedDecision, ok := request.ConditionallyAuthorizedDecisionFrom(ctx)
	if !ok {
		// In the unconditionally authorized path, nothing is added to the context, hence this means "directly authorized"
		return nil
	}

	authzAttrs, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get authorizer attributes: %w", err)
	}

	return EnforceConditions(ctx, a, o, authorizer, authzAttrs, unevaluatedDecision, c.builtinConditionSetEvaluators...)
}

// TODO(luxas): This function should probably be under the authorizer package, and then this concrete admission plugin just returns it?
func EnforceConditions(ctx context.Context, admissionAttrs admission.Attributes, o admission.ObjectInterfaces, authorizer authorizer.Authorizer, authzAttrs authorizer.Attributes, unevaluatedDecision authorizer.Decision, builtinEvaluators ...authorizer.BuiltinConditionSetEvaluator) error {
	// TODO: Does this convert to the request GVR version?
	versionedAttributes, err := admission.NewVersionedAttributes(admissionAttrs, admissionAttrs.GetKind(), o)
	if err != nil {
		return fmt.Errorf("failed to convert object version: %w", err)
	}

	data := conditionsData{
		attrsShim: attrsShim{
			VersionedAttributes: versionedAttributes,
		},
	}

	evaluatedDecision, err := EvaluateConditionsWithBuiltins(ctx, authorizer, unevaluatedDecision, data, builtinEvaluators...)
	// At this point, we require an unconditional allow in order to proceed.
	if evaluatedDecision.IsAllowed() {
		return nil
	}

	if err != nil {
		//audit.AddAuditAnnotation(ctx, reasonAnnotationKey, reasonError)
		return apierrors.NewInternalError(err) // TODO: Check if this is the same as responsewriters.InternalError(w, req, err)
	}

	reason := evaluatedDecision.Reason()
	klog.V(4).InfoS("Forbidden (after conditional authorization)", "URI", authzAttrs.GetPath(), "reason", reason)
	//audit.AddAuditAnnotations(ctx,
	//	decisionAnnotationKey, decisionForbid,
	//	reasonAnnotationKey, reason)

	return apierrors.NewForbidden(versionedAttributes.GetResource().GroupResource(), versionedAttributes.GetName(), responsewriters.ForbiddenStatusError(authzAttrs, reason))
}

func EvaluateConditionsWithBuiltins(ctx context.Context, authorizer authorizer.Authorizer, unevaluatedDecision authorizer.Decision, data authorizer.ConditionData, builtinEvaluators ...authorizer.BuiltinConditionSetEvaluator) (authorizer.Decision, error) {
	if unevaluatedDecision.IsConcrete() {
		return unevaluatedDecision, nil
	}

	possiblyEvaluatedUsingPlugins, fullyEvaluated, err := tryEvaluateUsingBuiltins(ctx, unevaluatedDecision, data, builtinEvaluators...)
	if unevaluatedDecision.IsConcrete() || fullyEvaluated {
		return possiblyEvaluatedUsingPlugins, err
	}

	// Defer to the authorizer for the rest of the evaluation
	// TODO(luxas): Make sure that evaluatedDecisionAfter > evaluatedDecision before returning
	return authorizer.EvaluateConditions(ctx, possiblyEvaluatedUsingPlugins, data)
}

func tryEvaluateUsingBuiltins(ctx context.Context, unevaluatedDecision authorizer.Decision, data authorizer.ConditionData, builtinEvaluators ...authorizer.BuiltinConditionSetEvaluator) (authorizer.Decision, bool, error) {
	if unevaluatedDecision.IsConcrete() {
		return unevaluatedDecision, true, nil
	}

	if unevaluatedDecision.IsConditionalChain() {
		var newDecisionChain authorizer.ConditionalDecisionChain
		errlist := []error{}
		// Recursively walk through the decision DAG in a depth-first manner.
		for i, unevaluatedSubDecision := range unevaluatedDecision.ConditionalChain() {
			evaluatedSubDecision, fullyEvaluated, err := tryEvaluateUsingBuiltins(ctx, unevaluatedSubDecision, data, builtinEvaluators...)
			if err != nil {
				errlist = append(errlist, err)
			}

			if evaluatedSubDecision.IsAllowed() || evaluatedSubDecision.IsDenied() {
				return evaluatedSubDecision, true, utilerrors.NewAggregate(errlist)
			}

			newDecisionChain = append(newDecisionChain, evaluatedSubDecision)
			if evaluatedSubDecision.IsNoOpinion() {
				continue
			}
			// Either Conditional or ConditionChain. If it is considered "fully evaluated",
			// then we're ok to return it as a success, otherwise we return a pruned prefix
			// of the original chain, now with possible more NoOpinions in the beginning
			// of the chain than before (until the i-th position).
			// TODO(luxas): Make sure that evaluatedDecisionAfter > evaluatedDecision before assigning
			if fullyEvaluated {
				return evaluatedSubDecision, true, utilerrors.NewAggregate(errlist)
			}

			// Preserve the tail of the chain if we exit greedy builtin evaluation early
			// TODO(luxas): Make a unit test for this
			if i+1 < len(unevaluatedDecision.ConditionalChain()) {
				newDecisionChain = append(newDecisionChain, unevaluatedDecision.ConditionalChain()[i+1:]...)
			}

			return authorizer.DecisionConditionalChain(newDecisionChain...), false, utilerrors.NewAggregate(errlist)
		}
		// To get here, all evaluated decisions must have been NoOpinions, which aggregate to NoOpinion
		return authorizer.DecisionNoOpinion(), true, utilerrors.NewAggregate(errlist)
	}

	// Otherwise, the decision is a ConditionSet. Try to evaluate it using the builtin ones.
	conditionSet := unevaluatedDecision.ConditionSet()
	errlist := []error{}
	for _, builtinEvaluator := range builtinEvaluators {
		possiblyEvaluatedDecision, err := builtinEvaluator.BuiltinEvaluateConditions(ctx, conditionSet, data)
		if err != nil {
			errlist = append(errlist, err)
		}

		// The builtin evaluator could not evaluate this conditionset, try the next one
		if possiblyEvaluatedDecision == nil {
			continue
		}

		if possiblyEvaluatedDecision.IsConditionalChain() {
			err = fmt.Errorf("builtin ConditionSet evaluator %T returned ConditionalChain, that is invalid behavior", builtinEvaluator)
			errlist = append(errlist, err)
			continue
		}

		// possiblyEvaluatedDecision is an Allow, Deny, NoOpinion or other ConditionSet (against another target),
		// which the evaluator claims is fully evaluated. We can thus return this resolved result.
		// TODO(luxas): Make sure that evaluatedDecisionAfter > evaluatedDecision before returning
		return *possiblyEvaluatedDecision, true, utilerrors.NewAggregate(errlist)
	}
	// None of the evaluators could (fully) evaluate the ConditionSet, thus return the original
	return unevaluatedDecision, false, utilerrors.NewAggregate(errlist)
}

type conditionsData struct {
	attrsShim
}

func (d conditionsData) WriteRequest() authorizer.WriteRequestConditionData {
	return &d.attrsShim
}

func (d conditionsData) ImpersonationRequest() authorizer.ImpersonationRequestConditionData {
	return nil
}

type attrsShim struct {
	*admission.VersionedAttributes
}

// TODO: Can the authorizer package depend on the admission package? If not, we need to add this cast.
func (u *attrsShim) GetOperation() string {
	return string(u.Attributes.GetOperation())
}
