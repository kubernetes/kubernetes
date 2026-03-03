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
	decisionAnnotationKey = "authorization.k8s.io/decision"
	reasonAnnotationKey   = "authorization.k8s.io/reason"

	// Annotation values set in advanced audit
	decisionAllow            = "allow"
	decisionConditionalAllow = "conditional_allow"
	decisionForbid           = "forbid"
	reasonError              = "internal error"
)

// TODO(luxas): Update tests for WithAuthorization

type recordAuthorizationMetricsFunc func(ctx context.Context, authorized authorizer.Decision, err error, authStart time.Time, authFinish time.Time)

// ConditionalAuthorizationRequestClassifier is a function that returns true if a request with the given attributes supports conditional authorization
type ConditionalAuthorizationRequestClassifier func(attrs authorizer.Attributes) bool

// WithAuthorization passes all authorized requests on to handler, and returns a forbidden error otherwise.
func WithAuthorization(hhandler http.Handler, auth authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return withAuthorization(hhandler, auth, s, recordAuthorizationMetrics, nil)
}

// WithAuthorizationAndConditionsSupport passes all authorized requests on to handler, and returns a forbidden error otherwise.
// If conditionalAuthzClassifier returns true, it also allows conditionally authorized requests through.
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

		// Specifically opt-in to Conditional Authorization for this Authorize call.
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
			attributes.(*authorizer.AttributesRecord).ConditionsMode = authorizer.ConditionsModeOptimized
		}

		authorized, err := a.Authorize(ctx, attributes)
		reason := authorized.Reason()

		authorizationFinish := time.Now()
		request.TrackAuthorizationLatency(ctx, authorizationFinish.Sub(authorizationStart))
		defer func() {
			metrics(ctx, authorized, err, authorizationStart, authorizationFinish)
		}()

		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if authorized.IsAllowed() {
			audit.AddAuditAnnotations(ctx,
				decisionAnnotationKey, decisionAllow,
				reasonAnnotationKey, reason)
			handler.ServeHTTP(w, req)
			return
		}
		// Only let conditionally authorized requests through when the feature gate is enabled, and for requests that could become allowed
		// when the conditions are evaluated. This means that there must be at least one Allow condition present.
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) && authorized.CanBecomeAllowed() {
			// Protect against authorizers that return conditional responses for request types that do not support conditional authorization
			// Only requests that are
			// a) subject to admission (write + connect requests), or
			// b) handled by an aggregated API server (which in turn is responsible for properly authorizing the request)
			// should be let through here.
			// TODO: How do we know what types skip admission (such as SAR)?
			if conditionalAuthzClassifier != nil && conditionalAuthzClassifier(attributes) {
				ctx = request.WithConditionallyAuthorizedDecision(ctx, a, authorized)
				req = req.WithContext(ctx)
				audit.AddAuditAnnotations(ctx,
					decisionAnnotationKey, decisionConditionalAllow,
					reasonAnnotationKey, reason)
				handler.ServeHTTP(w, req)
				return
			}
		}
		if err != nil {
			audit.AddAuditAnnotation(ctx, reasonAnnotationKey, reasonError)
			responsewriters.InternalError(w, req, err)
			return
		}

		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason)
		audit.AddAuditAnnotations(ctx,
			decisionAnnotationKey, decisionForbid,
			reasonAnnotationKey, reason)
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
