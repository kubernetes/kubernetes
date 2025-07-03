/*
Copyright 2025 The Kubernetes Authors.

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
	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/klog/v2"
	"net/http"
	"strings"
)

// WithContrainedImpersonation is a filter that will inspect and check requests that attempt to change the user.Info for their requests
func WithContrainedImpersonation(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		impersonationRequests, err := buildImpersonationRequests(req.Header)
		if err != nil {
			klog.V(4).Infof("%v", err)
			responsewriters.InternalError(w, req, err)
			return
		}
		if len(impersonationRequests) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		ctx := req.Context()

		// Get request attributes from the context.
		attrs, err := GetAuthorizerAttributes(ctx)
		if err != nil {
			responsewriters.InternalError(w, req, err)
		}
		if attrs.GetUser() == nil {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		requestorAttrs := attrs.(*authorizer.AttributesRecord)

		decision, reason, actingAsAttributes, newUser := authorizeConstrainedImpersonation(req, requestorAttrs, impersonationRequests, a)
		if decision == authorizer.DecisionAllow {
			handleRequestAfterImpersonation(handler, w, req, newUser)
			return
		}

		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)

		// fallback to use legacy impersonation
		legacyDecision, legacyReason, _, newUser := authorizeImpersonation(req, attrs.GetUser(), impersonationRequests, a)
		if legacyDecision != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", legacyReason, "err", err)
			responsewriters.Forbidden(ctx, actingAsAttributes, w, req, reason, s)
			return
		}

		handleRequestAfterImpersonation(handler, w, req, newUser)
	})
}

func authorizeConstrainedImpersonation(req *http.Request, requestorAttrs *authorizer.AttributesRecord, impersonationRequests []v1.ObjectReference, a authorizer.Authorizer) (authorizer.Decision, string, authorizer.Attributes, *user.DefaultInfo) {
	ctx := req.Context()

	// if groups are not specified, then we need to look them up differently depending on the type of user
	// if they are specified, then they are the authority (including the inclusion of system:authenticated/system:unauthenticated groups)
	groupsSpecified := len(req.Header[authenticationv1.ImpersonateGroupHeader]) > 0

	// make sure we're allowed to impersonate each thing we're requesting.  While we're iterating through, start building username
	// and group information
	username := ""
	groups := []string{}
	userExtra := map[string][]string{}
	uid := ""

	for _, impersonationRequest := range impersonationRequests {
		gvk := impersonationRequest.GetObjectKind().GroupVersionKind()

		actingAsAttributes := &authorizer.AttributesRecord{
			User: requestorAttrs.User,
			// group and version should always be authentication.k8s.io and v1
			APIGroup:        authenticationv1.SchemeGroupVersion.Group,
			APIVersion:      authenticationv1.SchemeGroupVersion.Version,
			Namespace:       impersonationRequest.Namespace,
			Name:            impersonationRequest.Name,
			ResourceRequest: true,
		}

		switch gvk.GroupKind() {
		case v1.SchemeGroupVersion.WithKind("ServiceAccount").GroupKind():
			actingAsAttributes.Resource = "serviceaccounts"
			actingAsAttributes.Verb = "impersonate:serviceaccount"
			username = serviceaccount.MakeUsername(impersonationRequest.Namespace, impersonationRequest.Name)
			if !groupsSpecified {
				// if groups aren't specified for a service account, we know the groups because its a fixed mapping.  Add them
				groups = serviceaccount.MakeGroupNames(impersonationRequest.Namespace)
			}

		case v1.SchemeGroupVersion.WithKind("User").GroupKind():
			// If the user has the prefix of "system:node", impersonate:node or impersonate:scheduled-node is checked
			// instead of impersonate:user-info
			if strings.HasPrefix(actingAsAttributes.Name, "system:node:") {
				decision, attr, reason := authorizeNodeImperonsation(ctx, requestorAttrs.User, actingAsAttributes, a, req.RequestURI)
				if decision != authorizer.DecisionAllow {
					return decision, reason, attr, nil
				}
			} else {
				actingAsAttributes.Verb = "impersonate:user-info"
				actingAsAttributes.Resource = "users"
			}
			username = impersonationRequest.Name

		case v1.SchemeGroupVersion.WithKind("Group").GroupKind():
			actingAsAttributes.Resource = "groups"
			actingAsAttributes.Verb = "impersonate:user-info"
			groups = append(groups, impersonationRequest.Name)

		case authenticationv1.SchemeGroupVersion.WithKind("UserExtra").GroupKind():
			extraKey := impersonationRequest.FieldPath
			extraValue := impersonationRequest.Name
			actingAsAttributes.Resource = "userextras"
			actingAsAttributes.Verb = "impersonate:user-info"
			actingAsAttributes.Subresource = extraKey
			userExtra[extraKey] = append(userExtra[extraKey], extraValue)

		case authenticationv1.SchemeGroupVersion.WithKind("UID").GroupKind():
			actingAsAttributes.Resource = "uids"
			actingAsAttributes.Verb = "impersonate:user-info"
			uid = impersonationRequest.Name
		}

		decision, reason, err := a.Authorize(ctx, actingAsAttributes)
		if err != nil || decision != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)
			return decision, reason, actingAsAttributes, nil
		}
	}

	// Prepend the impersonate-on prefix to the actual verb.
	requestorAttrs.Verb = "impersonate-on:" + requestorAttrs.Verb
	decision, reason, err := a.Authorize(ctx, requestorAttrs)
	if err != nil || decision != authorizer.DecisionAllow {
		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)
	}

	return decision, reason, requestorAttrs, &user.DefaultInfo{Name: username, Groups: groups, UID: uid, Extra: userExtra}
}

// authorizeNodeImperonsation authorizes the request impersonating node.
func authorizeNodeImperonsation(ctx context.Context, requestor user.Info, attr *authorizer.AttributesRecord, a authorizer.Authorizer, requestURI string) (authorizer.Decision, authorizer.Attributes, string) {
	attr.Name = strings.TrimPrefix(attr.Name, "system:node:")
	attr.Resource = "nodes"
	attr.Verb = "impersonate:node"

	// if the requestor is using a service account to impersonate the node it is running on,
	// 1. check the permission with verb impersonate:scheduled-node.
	// 2. If fails, fallback to check the permission with verb impersonate:node.
	if isScheduledNode(requestor, attr) {
		attr.Verb = "impersonate:scheduled-node"
		decision, reason, err := a.Authorize(ctx, attr)

		// if impersonate:scheduled-node check fails, fallback to check impersonate:node.
		if err != nil || decision != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "URI", requestURI, "reason", reason, "err", err)
		} else {
			return decision, attr, reason
		}
	}

	decision, reason, err := a.Authorize(ctx, attr)
	if err != nil || decision != authorizer.DecisionAllow {
		klog.V(4).InfoS("Forbidden", "URI", requestURI, "reason", reason, "err", err)
	}
	return decision, attr, reason
}

// isScheduledNode checks if the requestor is from the scheduled node:
// 1. the requestor is impersonating a node.
// 2. the requestor must be using a service account.
// 3. the requestor must run on the same node it is impersonating.
func isScheduledNode(requestor user.Info, attr *authorizer.AttributesRecord) bool {
	if attr.Resource != "nodes" {
		return false
	}

	if _, _, err := serviceaccount.SplitUsername(requestor.GetName()); err != nil {
		return false
	}

	if len(requestor.GetExtra()) == 0 {
		return false
	}

	if len(requestor.GetExtra()[serviceaccount.NodeNameKey]) != 1 || requestor.GetExtra()[serviceaccount.NodeNameKey][0] != attr.GetName() {
		return false
	}

	return true
}
