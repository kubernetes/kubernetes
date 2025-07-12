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

// WithConstrainedImpersonation is a filter that will inspect and check requests that attempt to change the user.Info for their requests
func WithConstrainedImpersonation(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestor, exists := request.UserFrom(ctx)
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		actingAsAttrsList, newUser, err := buildImpersonationAttributes(req.Header, requestor)
		if err != nil {
			responsewriters.InternalError(w, req, err)
			return
		}
		if len(actingAsAttrsList) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		decision, reason, actingAsAttributes, err := authorizeConstrainedImpersonation(ctx, actingAsAttrsList, a)
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

		req = req.WithContext(request.WithUser(ctx, newUser))
		httplog.LogOf(req, w).Addf("%v is impersonating %v", userString(requestor), userString(newUser))

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

func buildImpersonationAttributes(headers http.Header, requestor user.Info) ([]*authorizer.AttributesRecord, user.Info, error) {
	attrsList := []*authorizer.AttributesRecord{}
	requestedUser := headers.Get(authenticationv1.ImpersonateUserHeader)
	groups := headers[authenticationv1.ImpersonateGroupHeader]
	userExtra := map[string][]string{}
	uid := headers.Get(authenticationv1.ImpersonateUIDHeader)

	hasUser := len(requestedUser) > 0
	if hasUser {
		attrs := newAttributeRecord(requestor)
		if namespace, name, err := serviceaccount.SplitUsername(requestedUser); err == nil {
			attrs.Resource = "serviceaccounts"
			attrs.Verb = "impersonate:serviceaccount"
			attrs.Namespace = namespace
			attrs.Name = name
			attrsList = append(attrsList, attrs)
		} else if isScheduledNode(requestor, requestedUser) {
			attrs.Verb = "impersonate:scheduled-node"
			attrs.Name = strings.TrimPrefix(requestedUser, "system:node:")
			attrs.Resource = "nodes"
			attrsList = append(attrsList, attrs)
		} else if strings.HasPrefix(requestedUser, "system:node:") {
			attrs.Verb = "impersonate:node"
			attrs.Name = strings.TrimPrefix(requestedUser, "system:node:")
			attrs.Resource = "nodes"
			attrsList = append(attrsList, attrs)
		} else {
			attrs.Verb = "impersonate:user-info"
			attrs.Name = requestedUser
			attrs.Resource = "users"
			attrsList = append(attrsList, attrs)
		}
	}

	// if groups are not specified, then we need to look them up differently depending on the type of user
	// if they are specified, then they are the authority (including the inclusion of system:authenticated/system:unauthenticated groups)
	hasGroups := len(groups) > 0
	for _, group := range groups {
		attrs := newAttributeRecord(requestor)
		attrs.Resource = "groups"
		attrs.Name = group
		attrs.Verb = "impersonate:user-info"
		attrsList = append(attrsList, attrs)
	}

	hasUserExtra := false
	for headerName, values := range headers {
		if !strings.HasPrefix(headerName, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
			continue
		}

		hasUserExtra = true
		extraKey := unescapeExtraKey(strings.ToLower(headerName[len(authenticationv1.ImpersonateUserExtraHeaderPrefix):]))

		// make a separate request for each extra value they're trying to set
		for _, value := range values {
			attrs := newAttributeRecord(requestor)
			attrs.Verb = "impersonate:user-info"
			attrs.Name = value
			attrs.Subresource = extraKey
			attrs.Resource = "userextras"
			attrsList = append(attrsList, attrs)
			userExtra[extraKey] = append(userExtra[extraKey], value)
		}
	}

	hasUID := len(uid) > 0
	if hasUID {
		attrs := newAttributeRecord(requestor)
		attrs.Verb = "impersonate:user-info"
		attrs.Resource = "uids"
		attrs.Name = uid
		attrsList = append(attrsList, attrs)
	}

	if (hasGroups || hasUserExtra || hasUID) && !hasUser {
		return nil, nil, fmt.Errorf("requested %v without impersonating a user", attrsList)
	}

	// If the user is service account and groups is empty, set default service account groups.
	if namespace, _, err := serviceaccount.SplitUsername(requestedUser); err == nil {
		if !hasGroups {
			groups = serviceaccount.MakeGroupNames(namespace)
		}
	}

	if requestedUser != user.Anonymous {
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
		Name:   requestedUser,
		Groups: groups,
		Extra:  userExtra,
		UID:    uid,
	}

	return attrsList, newUser, nil
}

func newAttributeRecord(requestor user.Info) *authorizer.AttributesRecord {
	return &authorizer.AttributesRecord{
		APIGroup:        authenticationv1.SchemeGroupVersion.Group,
		APIVersion:      authenticationv1.SchemeGroupVersion.Version,
		User:            requestor,
		ResourceRequest: true,
	}
}

func authorizeConstrainedImpersonation(ctx context.Context, targetAttributesList []*authorizer.AttributesRecord, a authorizer.Authorizer) (authorizer.Decision, string, authorizer.Attributes, error) {
	// Get request attributes from the context.
	requestorAttrs, err := GetAuthorizerAttributes(ctx)
	if err != nil {
		return authorizer.DecisionNoOpinion, "", &authorizer.AttributesRecord{}, err
	}

	for _, actingAsAttributes := range targetAttributesList {
		decision, reason, err := a.Authorize(ctx, actingAsAttributes)

		if decision == authorizer.DecisionAllow {
			continue
		}

		// Fallback to authorize impersonate:node
		if actingAsAttributes.Verb == "impersonate:scheduled-node" {
			attrCopy := copyAuthorizerAttr(actingAsAttributes)
			attrCopy.Verb = "impersonate:node"
			decision, reason, err = a.Authorize(ctx, attrCopy)
		}

		if err != nil || decision != authorizer.DecisionAllow {
			return decision, reason, actingAsAttributes, err
		}
	}

	// Prepend the impersonate-on prefix to the actual verb.
	impersonateOnAttrs := copyAuthorizerAttr(requestorAttrs)
	impersonateOnAttrs.Verb = fmt.Sprintf("impersonate-on:%s", impersonateOnAttrs.Verb)
	decision, reason, err := a.Authorize(ctx, impersonateOnAttrs)
	return decision, reason, impersonateOnAttrs, err
}

func copyAuthorizerAttr(attr authorizer.Attributes) *authorizer.AttributesRecord {
	out := &authorizer.AttributesRecord{
		APIGroup:        attr.GetAPIGroup(),
		APIVersion:      attr.GetAPIVersion(),
		User:            attr.GetUser(),
		Verb:            attr.GetVerb(),
		Resource:        attr.GetResource(),
		Subresource:     attr.GetSubresource(),
		Name:            attr.GetName(),
		Namespace:       attr.GetNamespace(),
		ResourceRequest: attr.IsResourceRequest(),
		Path:            attr.GetPath(),
	}

	out.LabelSelectorRequirements, out.LabelSelectorParsingErr = attr.GetLabelSelector()
	out.FieldSelectorRequirements, out.FieldSelectorParsingErr = attr.GetFieldSelector()
	return out
}

func authorizeLegacyImpersonation(req *http.Request, targetAttributesList []*authorizer.AttributesRecord, a authorizer.Authorizer) (authorizer.Decision, string, error) {
	ctx := req.Context()

	for _, actingAsAttributes := range targetAttributesList {
		// verb is always impersonate with legacy impersonation.
		actingAsAttributesCopy := copyAuthorizerAttr(actingAsAttributes)
		actingAsAttributesCopy.Verb = "impersonate"

		// legacy impersonation does not recognize nodes, change back to users
		if actingAsAttributesCopy.Resource == "nodes" {
			actingAsAttributesCopy.Resource = "users"
			actingAsAttributesCopy.Name = fmt.Sprintf("system:node:%s", actingAsAttributesCopy.Name)
		}

		// group and version is empty for users/groups/serviceaccounts
		if actingAsAttributesCopy.Resource == "users" || actingAsAttributesCopy.Resource == "groups" || actingAsAttributesCopy.Resource == "serviceaccounts" {
			actingAsAttributesCopy.APIGroup = ""
			actingAsAttributesCopy.APIVersion = ""
		}

		decision, reason, err := a.Authorize(ctx, actingAsAttributesCopy)
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
func isScheduledNode(requestor user.Info, impersonatedUser string) bool {
	if !strings.HasPrefix(impersonatedUser, "system:node:") {
		return false
	}
	nodeName := strings.TrimPrefix(impersonatedUser, "system:node:")

	if _, _, err := serviceaccount.SplitUsername(requestor.GetName()); err != nil {
		return false
	}

	if len(requestor.GetExtra()) == 0 {
		return false
	}

	if len(requestor.GetExtra()[serviceaccount.NodeNameKey]) != 1 || requestor.GetExtra()[serviceaccount.NodeNameKey][0] != nodeName {
		return false
	}

	return true
}
