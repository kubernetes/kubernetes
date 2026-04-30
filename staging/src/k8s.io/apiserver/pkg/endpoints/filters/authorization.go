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
	"k8s.io/utils/ptr"

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
	DecisionAnnotationKey = "authorization.k8s.io/decision"
	ReasonAnnotationKey   = "authorization.k8s.io/reason"

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
func WithAuthorization(hhandler http.Handler, auth authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return withAuthorization(hhandler, auth, s, recordAuthorizationMetrics, nil)
}

// WithAuthorizationAndConditionsSupport passes all authorized requests on to handler, and returns a forbidden error otherwise.
// If conditionalAuthzClassifier returns true, it also allows conditionally authorized requests through, as then the classifier
// guarantees that there is some conditions enforcement later in the request handler chain.
func WithAuthorizationAndConditionsSupport(hhandler http.Handler, auth authorizer.Authorizer, s runtime.NegotiatedSerializer, conditionalAuthzClassifier ConditionalAuthorizationRequestClassifier) http.Handler {
	return withAuthorization(hhandler, auth, s, recordAuthorizationMetrics, conditionalAuthzClassifier)
}

func withAuthorization(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer, metrics recordAuthorizationMetricsFunc, conditionalAuthzClassifier ConditionalAuthorizationRequestClassifier) http.Handler {
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

		var reason string
		var metricsResultLabel string
		var unconditionallyAuthorized bool // = false, unless set in the branches below
		var conditionsAwareDecision *authorizer.ConditionsAwareDecision

		// Both branches must set unconditionallyAuthorized, reason, err, and metricsResultLabel properly
		// Only call ConditionsAwareAuthorize for requests that support conditional authorization.
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) && conditionalAuthzClassifier != nil && conditionalAuthzClassifier(attributes) {
			conditionsAwareDecision = ptr.To(a.ConditionsAwareAuthorize(ctx, attributes))

			unconditionallyAuthorized = conditionsAwareDecision.IsAllowed()
			reason = conditionsAwareDecision.Reason()
			err = conditionsAwareDecision.Error()
			metricsResultLabel = authorizationMetricsLabelForAuthorizeConditionsAware(*conditionsAwareDecision)
		} else {
			var decision authorizer.Decision
			decision, reason, err = a.Authorize(ctx, attributes)

			unconditionallyAuthorized = decision == authorizer.DecisionAllow
			// reason, err are already set above
			metricsResultLabel = authorizationMetricsLabelForAuthorize(decision, err)
		}

		authorizationFinish := time.Now()
		request.TrackAuthorizationLatency(ctx, authorizationFinish.Sub(authorizationStart))
		defer func() {
			metrics(ctx, metricsResultLabel, authorizationStart, authorizationFinish)
		}()

		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if unconditionallyAuthorized {
			audit.AddAuditAnnotations(ctx,
				DecisionAnnotationKey, DecisionAllow,
				ReasonAnnotationKey, reason)
			handler.ServeHTTP(w, req)
			return
		}

		// Only let conditionally-authorized requests proceed when:
		// a) the conditional authorization feature is enabled
		// b) there is a possibility that the conditional decision can evaluate to Allow (otherwise, fail fast)
		// c) if this request now is let past this filter, there exists some later HTTP handler in the chain which
		//	  evaluates the conditions against the admission control data, and enforces that the decision evaluates to Allow
		//    in order to let the request through to the storage layer. This is pluggably decided by conditionalAuthzClassifier.
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) && conditionsAwareDecision != nil {
			if conditionsAwareDecision.CanBecomeAllowed() {
				ctx = request.WithConditionallyAuthorizedDecision(ctx, a, *conditionsAwareDecision)
				req = req.WithContext(ctx)
				// Audit annotations are set in the AuthorizationConditionsEnforcer admission plugin.
				handler.ServeHTTP(w, req)
				return
			}
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

	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
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
	}

	return &attribs, nil
}
