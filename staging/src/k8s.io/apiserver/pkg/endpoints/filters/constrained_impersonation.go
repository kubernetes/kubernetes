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
	"fmt"
	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
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

		// if groups are not specified, then we need to look them up differently depending on the type of user
		// if they are specified, then they are the authority (including the inclusion of system:authenticated/system:unauthenticated groups)
		groupsSpecified := len(req.Header[authenticationv1.ImpersonateGroupHeader]) > 0

		// Building attributes to authorize, username and group information
		var actingAsAttrsList []*authorizer.AttributesRecord
		username := ""
		groups := []string{}
		userExtra := map[string][]string{}
		uid := ""
		for _, impersonationRequest := range impersonationRequests {
			gvk := impersonationRequest.GetObjectKind().GroupVersionKind()
			actingAsAttributes := &authorizer.AttributesRecord{
				User:            attrs.GetUser(),
				APIGroup:        gvk.Group,
				APIVersion:      gvk.Version,
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
				actingAsAttributes.Resource = "users"
				actingAsAttributes.Verb = "impersonate:user-info"
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
				uid = impersonationRequest.Name
				actingAsAttributes.Resource = "uids"
				actingAsAttributes.Verb = "impersonate:user-info"

			default:
				klog.V(4).InfoS("unknown impersonation request type", "request", impersonationRequest)
				responsewriters.Forbidden(ctx, actingAsAttributes, w, req, fmt.Sprintf("unknown impersonation request type: %v", impersonationRequest), s)
				return
			}

			actingAsAttrsList = append(actingAsAttrsList, actingAsAttributes)
		}

		decision, reason, actingAsAttributes, err := authorizeConstrainedImpersonation(ctx, requestorAttrs, actingAsAttrsList, a)
		if err != nil || decision != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", reason, "err", err)
			// fallback to use legacy impersonation
			legacyDecision, legacyReason, err := authorizeLegacyImpersonation(req, actingAsAttrsList, a)
			if err != nil || legacyDecision != authorizer.DecisionAllow {
				klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "reason", legacyReason, "err", err)
				responsewriters.Forbidden(ctx, actingAsAttributes, w, req, reason, s)
				return
			}
		}

		if username != user.Anonymous {
			// When impersonating a non-anonymous user, include the 'system:authenticated' group
			// in the impersonated user info:
			// - if no groups were specified
			// - if a group has been specified other than 'system:authenticated'
			//
			// If 'system:unauthenticated' group has been specified we should not include
			// the 'system:authenticated' group.
			addAuthenticated := true
			for _, group := range groups {
				if group == user.AllAuthenticated || group == user.AllUnauthenticated {
					addAuthenticated = false
					break
				}
			}

			if addAuthenticated {
				groups = append(groups, user.AllAuthenticated)
			}
		} else {
			addUnauthenticated := true
			for _, group := range groups {
				if group == user.AllUnauthenticated {
					addUnauthenticated = false
					break
				}
			}

			if addUnauthenticated {
				groups = append(groups, user.AllUnauthenticated)
			}
		}

		newUser := &user.DefaultInfo{
			Name:   username,
			Groups: groups,
			Extra:  userExtra,
			UID:    uid,
		}
		req = req.WithContext(request.WithUser(ctx, newUser))

		oldUser, _ := request.UserFrom(ctx)
		httplog.LogOf(req, w).Addf("%v is impersonating %v", userString(oldUser), userString(newUser))

		audit.LogImpersonatedUser(audit.WithAuditContext(ctx), newUser)

		// clear all the impersonation headers from the request
		req.Header.Del(authenticationv1.ImpersonateUserHeader)
		req.Header.Del(authenticationv1.ImpersonateGroupHeader)
		req.Header.Del(authenticationv1.ImpersonateUIDHeader)
		for headerName := range req.Header {
			if strings.HasPrefix(headerName, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
				req.Header.Del(headerName)
			}
		}

		handler.ServeHTTP(w, req)
	})
}

func authorizeConstrainedImpersonation(ctx context.Context, requestorAttrs *authorizer.AttributesRecord, targetAttributesList []*authorizer.AttributesRecord, a authorizer.Authorizer) (authorizer.Decision, string, authorizer.Attributes, error) {
	for _, actingAsAttributes := range targetAttributesList {
		// group and version are always authentication/v1 with constrained impersonation.
		actingAsAttributesCopy := *actingAsAttributes
		actingAsAttributesCopy.APIGroup = authenticationv1.SchemeGroupVersion.Group
		actingAsAttributesCopy.APIVersion = authenticationv1.SchemeGroupVersion.Version

		// authorize node verb if the impersonated user is node
		if actingAsAttributes.Resource == "users" && strings.HasPrefix(actingAsAttributesCopy.Name, "system:node:") {
			decision, reason, attrs, err := authorizeNodeImperonsation(ctx, requestorAttrs.User, &actingAsAttributesCopy, a)
			if err != nil || decision != authorizer.DecisionAllow {
				return decision, reason, attrs, err
			}
			continue
		}

		decision, reason, err := a.Authorize(ctx, &actingAsAttributesCopy)
		if err != nil || decision != authorizer.DecisionAllow {
			return decision, reason, &actingAsAttributesCopy, err
		}
	}

	// Prepend the impersonate-on prefix to the actual verb.
	requestorAttrs.Verb = "impersonate-on:" + requestorAttrs.Verb
	decision, reason, err := a.Authorize(ctx, requestorAttrs)
	return decision, reason, requestorAttrs, err
}

// authorizeNodeImperonsation authorizes the request impersonating node.
func authorizeNodeImperonsation(ctx context.Context, requestor user.Info, attr *authorizer.AttributesRecord, a authorizer.Authorizer) (authorizer.Decision, string, authorizer.Attributes, error) {
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
		if decision != authorizer.DecisionAllow {
			klog.V(4).InfoS("Forbidden", "reason", reason, "err", err)
		} else {
			return decision, reason, attr, nil
		}
	}

	decision, reason, err := a.Authorize(ctx, attr)
	return decision, reason, attr, err
}

func authorizeLegacyImpersonation(req *http.Request, targetAttributesList []*authorizer.AttributesRecord, a authorizer.Authorizer) (authorizer.Decision, string, error) {
	ctx := req.Context()

	for _, actingAsAttributes := range targetAttributesList {
		// verb is always impersonate with legacy impersonation.
		actingAsAttributesCopy := *actingAsAttributes
		actingAsAttributesCopy.Verb = "impersonate"
		decision, reason, err := a.Authorize(ctx, &actingAsAttributesCopy)
		if err != nil || decision != authorizer.DecisionAllow {
			return decision, reason, err
		}
	}

	return authorizer.DecisionAllow, "", nil
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
