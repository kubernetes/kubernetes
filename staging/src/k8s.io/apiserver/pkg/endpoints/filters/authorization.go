/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"context"
	"errors"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

const (
	// Annotation key names set in advanced audit
	// These are public, to allow usage in the authorization conditions enforcer.
	DecisionAnnotationKey         = "authorization.k8s.io/decision"
	ReasonAnnotationKey           = "authorization.k8s.io/reason"
	isConditionalAuthorizationKey = "authorization.k8s.io/is-conditional-decision"

	// Annotation values set in advanced audit
	DecisionAllow  = "allow"
	DecisionForbid = "forbid"
	ReasonError    = "internal error"
)

// ConditionalAuthorizationRequestClassifier is a function that returns true if a request with the given attributes supports conditional authorization.
// If the function returns true, it MUST guarantee that there is some conditions enforcement later in the request handler chain.
type ConditionalAuthorizationRequestClassifier func(attrs authorizer.Attributes) bool

type recordAuthorizationMetricsFunc func(ctx context.Context, resultLabel string, authStart time.Time, authFinish time.Time)

// WithAuthorization passes all authorized requests on to handler, and returns a forbidden error otherwise.
func WithAuthorization(hhandler http.Handler, auth authorizer.UnconditionalAuthorizer, s runtime.NegotiatedSerializer) http.Handler {
	RegisterMetrics()
	return withAuthorization(hhandler, auth, s, recordAuthorizationMetrics)
}

func withAuthorization(handler http.Handler, a authorizer.UnconditionalAuthorizer, s runtime.NegotiatedSerializer, metrics recordAuthorizationMetricsFunc) http.Handler {
	if a == nil {
		klog.Warning("Authorization is disabled")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		authorizationStart := time.Now()

		attributes, err := GetAuthorizerAttributes(ctx)
		if err != nil {
			responsewriters.InternalError(w, req, err)
			return
		}
		authorized, reason, err := a.Authorize(ctx, attributes)

		metricsResultLabel := authorizationMetricsLabelForAuthorize(authorized, err)

		authorizationFinish := time.Now()
		request.TrackAuthorizationLatency(ctx, authorizationFinish.Sub(authorizationStart))
		defer func() {
			metrics(ctx, metricsResultLabel, authorizationStart, authorizationFinish)
		}()

		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if authorized == authorizer.DecisionAllow {
			audit.AddAuditAnnotations(ctx,
				DecisionAnnotationKey, DecisionAllow,
				ReasonAnnotationKey, reason)
			handler.ServeHTTP(w, req)
			return
		}
		if err != nil {
			audit.AddAuditAnnotation(ctx, ReasonAnnotationKey, ReasonError)
			responsewriters.InternalError(w, req, err)
			return
		}

		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason)
		audit.AddAuditAnnotations(ctx,
			DecisionAnnotationKey, DecisionForbid,
			ReasonAnnotationKey, reason)
		responsewriters.Forbidden(attributes, w, req, reason, s)
	})
}

// WithConditionsAwareAuthorization passes all authorized requests on to handler, and returns a forbidden error otherwise.
// If conditionalAuthzClassifier returns true, it also allows conditionally authorized requests through, as then the classifier
// guarantees that there is some conditions enforcement later in the request handler chain.
//
// When the ConditionalAuthorization feature gate is on AND the AuthorizationConditionsEnforcer admission plugin is enabled,
// the request always flows through the conditions-aware handler so a ConditionsAwareDecision is attached to the context —
// the enforcer plugin requires that context value to be present even when the classifier says "no conditional support here"
// (it then observes an unconditional decision and short-circuits). Only when the gate is off or the plugin is disabled
// do we fall back to the plain WithAuthorization filter, which is the pre-feature behaviour.
func WithConditionsAwareAuthorization(hhandler http.Handler, auth authorizer.Authorizer, s runtime.NegotiatedSerializer, conditionsEnforcerEnabled bool, conditionalAuthzClassifier ConditionalAuthorizationRequestClassifier) http.Handler {
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) && conditionsEnforcerEnabled && conditionalAuthzClassifier != nil {
		RegisterMetrics()
		return withConditionsAwareAuthorization(hhandler, auth, s, recordAuthorizationMetrics, conditionalAuthzClassifier)
	}
	return WithAuthorization(hhandler, auth, s)
}

func withConditionsAwareAuthorization(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer, metrics recordAuthorizationMetricsFunc, conditionalAuthzClassifier ConditionalAuthorizationRequestClassifier) http.Handler {
	if a == nil {
		klog.Warning("Authorization is disabled")
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		authorizationStart := time.Now()

		attributes, err := GetAuthorizerAttributes(ctx)
		if err != nil {
			responsewriters.InternalError(w, req, err)
			return
		}

		var conditionsAwareDecision authorizer.ConditionsAwareDecision

		// Both branches must set unconditionallyAuthorized, reason, err, and metricsResultLabel properly
		// Only call ConditionsAwareAuthorize for requests that support conditional authorization.
		if conditionalAuthzClassifier != nil && conditionalAuthzClassifier(attributes) {
			conditionsAwareDecision = a.ConditionsAwareAuthorize(ctx, attributes)
		} else {
			conditionsAwareDecision = authorizer.ConditionsAwareDecisionFromParts(a.Authorize(ctx, attributes))
		}

		// Attach the conditionsAwareDecision to the request context
		ctx = request.WithConditionallyAuthorizedDecision(ctx, a, conditionsAwareDecision)
		req = req.WithContext(ctx)

		metricsResultLabel := authorizationMetricsLabelForAuthorizeConditionsAware(conditionsAwareDecision)

		authorizationFinish := time.Now()
		request.TrackAuthorizationLatency(ctx, authorizationFinish.Sub(authorizationStart))
		defer func() {
			metrics(ctx, metricsResultLabel, authorizationStart, authorizationFinish)
		}()

		// Set reason and err variables to keep parity with withAuthorization above
		reason := conditionsAwareDecision.Reason()
		err = conditionsAwareDecision.Error()

		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if conditionsAwareDecision.IsAllow() {
			// Set the audit annotation already here, as the request might fail before reaching
			// the AuthorizationConditionsEnforcer validating admission plugin
			audit.AddAuditAnnotations(ctx,
				DecisionAnnotationKey, DecisionAllow,
				ReasonAnnotationKey, reason)

			handler.ServeHTTP(w, req)
			return
		}

		// If the request is conditionally allowed, proceed to the AuthorizationConditionsEnforcer and let it enforce the final decision.
		if conditionsAwareDecision.PossibleDecisions().Has(authorizer.DecisionAllow) {
			// Mark conditional requests in audit logs, especially for cases where the request would fail before reaching
			// the AuthorizationConditionsEnforcer validating admission plugin
			audit.AddAuditAnnotation(ctx, isConditionalAuthorizationKey, "true")

			handler.ServeHTTP(w, req)
			return
		}

		if err != nil {
			audit.AddAuditAnnotation(ctx, ReasonAnnotationKey, ReasonError)
			responsewriters.InternalError(w, req, err)
			return
		}

		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason)
		audit.AddAuditAnnotations(ctx,
			DecisionAnnotationKey, DecisionForbid,
			ReasonAnnotationKey, reason)
		responsewriters.Forbidden(attributes, w, req, reason, s)
	})
}

func GetAuthorizerAttributes(ctx context.Context) (authorizer.Attributes, error) {
	attribs := authorizer.AttributesRecord{}

	user, ok := request.UserFrom(ctx)
	if ok {
		attribs.User = user
	}

	requestInfo, found := request.RequestInfoFrom(ctx)
	if !found {
		return nil, errors.New("no RequestInfo found in the context")
	}

	// Start with common attributes that apply to resource and non-resource requests
	attribs.ResourceRequest = requestInfo.IsResourceRequest
	attribs.Path = requestInfo.Path
	attribs.Verb = requestInfo.Verb

	attribs.APIGroup = requestInfo.APIGroup
	attribs.APIVersion = requestInfo.APIVersion
	attribs.Resource = requestInfo.Resource
	attribs.Subresource = requestInfo.Subresource
	attribs.Namespace = requestInfo.Namespace
	attribs.Name = requestInfo.Name

	// parsing here makes it easy to keep the AttributesRecord type value-only and avoids any mutex copies when
	// doing shallow copies in other steps.
	if len(requestInfo.FieldSelector) > 0 {
		fieldSelector, err := fields.ParseSelector(requestInfo.FieldSelector)
		if err != nil {
			attribs.FieldSelectorRequirements, attribs.FieldSelectorParsingErr = nil, err
		} else {
			if requirements := fieldSelector.Requirements(); len(requirements) > 0 {
				attribs.FieldSelectorRequirements, attribs.FieldSelectorParsingErr = fieldSelector.Requirements(), nil
			}
		}
	}

	if len(requestInfo.LabelSelector) > 0 {
		labelSelector, err := labels.Parse(requestInfo.LabelSelector)
		if err != nil {
			attribs.LabelSelectorRequirements, attribs.LabelSelectorParsingErr = nil, err
		} else {
			if requirements, _ /*selectable*/ := labelSelector.Requirements(); len(requirements) > 0 {
				attribs.LabelSelectorRequirements, attribs.LabelSelectorParsingErr = requirements, nil
			}
		}
	}

	return &attribs, nil
}
