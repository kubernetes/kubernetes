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

const (
	legacyImpersonateVerb        = "impersonate"
	impersonateUserInfoVerb      = "impersonate:user-info"
	impersonateSAVerb            = "impersonate:serviceaccount"
	impersonateNodeVerb          = "impersonate:node"
	impersonateScheduledNodeVerb = "impersonate:scheduled-node"
	impersonateOnVerbPrefix      = "impersonate-on:"
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

		impersonateVerb := legacyImpersonateVerb
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
		} else {
			impersonateVerb = actingAsAttributes.GetVerb()
		}

		req = req.WithContext(request.WithUser(ctx, newUser))
		httplog.LogOf(req, w).Addf("%v is impersonating %v", userString(requestor), userString(newUser))

		audit.LogConstrainedImpersonateUser(audit.WithAuditContext(ctx), newUser, impersonateVerb)

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
	hasGroups := len(groups) > 0
	if hasUser {
		attrs := newAttributeRecord(requestor)
		if namespace, name, err := serviceaccount.SplitUsername(requestedUser); err == nil {
			attrs.Resource = "serviceaccounts"
			attrs.Verb = impersonateSAVerb
			attrs.Namespace = namespace
			attrs.Name = name
		} else if nodeName, isImpersonating := isImpersonatingScheduledNode(requestor, requestedUser, groups); isImpersonating {
			attrs.Verb = impersonateScheduledNodeVerb
			attrs.Name = nodeName
			attrs.Resource = "nodes"
		} else if nodeName, isImpersonating := isImpersonatingNode(requestedUser, groups); isImpersonating {
			attrs.Verb = impersonateNodeVerb
			attrs.Name = nodeName
			attrs.Resource = "nodes"
		} else {
			attrs.Verb = impersonateUserInfoVerb
			attrs.Name = requestedUser
			attrs.Resource = "users"
		}
		attrsList = append(attrsList, attrs)
	}

	// if the groups are specified, then they are the authority (including the inclusion of system:authenticated/system:unauthenticated groups)
	for _, group := range groups {
		// skip the node group of the user is impersonating a node.
		if _, isNode := isImpersonatingNode(requestedUser, groups); isNode && group == user.NodesGroup {
			continue
		}
		attrs := newAttributeRecord(requestor)
		attrs.Resource = "groups"
		attrs.Name = group
		attrs.Verb = impersonateUserInfoVerb
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
			attrs.Verb = impersonateUserInfoVerb
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
		attrs.Verb = impersonateUserInfoVerb
		attrs.Resource = "uids"
		attrs.Name = uid
		attrsList = append(attrsList, attrs)
	}

	// If the user is service account and groups is empty, set default service account groups.
	if namespace, _, err := serviceaccount.SplitUsername(requestedUser); err == nil {
		if !hasGroups {
			groups = serviceaccount.MakeGroupNames(namespace)
		}
	}

	if (hasGroups || hasUserExtra || hasUID) && !hasUser {
		return nil, nil, fmt.Errorf("requested %v without impersonating a user", attrsList)
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
		if actingAsAttributes.Verb == impersonateScheduledNodeVerb {
			attrCopy := copyAuthorizerAttr(actingAsAttributes)
			attrCopy.Verb = impersonateNodeVerb
			decision, reason, err = a.Authorize(ctx, attrCopy)
		}

		if err != nil || decision != authorizer.DecisionAllow {
			return decision, reason, actingAsAttributes, err
		}
	}

	// Prepend the impersonate-on prefix to the actual verb.
	impersonateOnAttrs := copyAuthorizerAttr(requestorAttrs)
	impersonateOnAttrs.Verb = fmt.Sprintf("%s%s", impersonateOnVerbPrefix, impersonateOnAttrs.Verb)
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

	var legacyActingAsAttributes []*authorizer.AttributesRecord
	for _, actingAsAttributes := range targetAttributesList {
		// verb is always impersonate with legacy impersonation.
		actingAsAttributesCopy := copyAuthorizerAttr(actingAsAttributes)
		actingAsAttributesCopy.Verb = legacyImpersonateVerb

		// legacy impersonation does not recognize nodes, change back to users
		// Also add node groups attr to be authorized.
		if actingAsAttributesCopy.Resource == "nodes" {
			actingAsAttributesCopy.Resource = "users"
			actingAsAttributesCopy.Name = fmt.Sprintf("system:node:%s", actingAsAttributesCopy.Name)
			nodeGroupAtrribute := &authorizer.AttributesRecord{
				Resource:        "groups",
				Verb:            legacyImpersonateVerb,
				Name:            user.NodesGroup,
				User:            actingAsAttributes.User,
				ResourceRequest: true,
			}
			legacyActingAsAttributes = append(legacyActingAsAttributes, nodeGroupAtrribute)
		}

		// group and version is empty for users/groups/serviceaccounts
		if actingAsAttributesCopy.Resource == "users" || actingAsAttributesCopy.Resource == "groups" || actingAsAttributesCopy.Resource == "serviceaccounts" {
			actingAsAttributesCopy.APIGroup = ""
			actingAsAttributesCopy.APIVersion = ""
		}

		legacyActingAsAttributes = append(legacyActingAsAttributes, actingAsAttributesCopy)
	}

	for _, actingAsAttributes := range legacyActingAsAttributes {
		decision, reason, err := a.Authorize(ctx, actingAsAttributes)
		if err != nil || decision != authorizer.DecisionAllow {
			return decision, reason, err
		}
	}

	return authorizer.DecisionAllow, "", nil
}

func isImpersonatingNode(impersonatedUser string, groups []string) (string, bool) {
	if !strings.HasPrefix(impersonatedUser, "system:node:") {
		return "", false
	}

	var hasNodeGroup bool
	for _, group := range groups {
		if group == user.NodesGroup {
			hasNodeGroup = true
		}
	}

	if !hasNodeGroup {
		return "", false
	}

	return strings.TrimPrefix(impersonatedUser, "system:node:"), true
}

// isScheduledNode checks if the requestor is from the scheduled node:
// 1. the requestor is impersonating a node.
// 2. the requestor must be using a service account.
// 3. the requestor must run on the same node it is impersonating.
func isImpersonatingScheduledNode(requestor user.Info, impersonatedUser string, groups []string) (string, bool) {
	nodeName, isNode := isImpersonatingNode(impersonatedUser, groups)
	if !isNode {
		return "", false
	}

	if _, _, err := serviceaccount.SplitUsername(requestor.GetName()); err != nil {
		return "", false
	}

	if len(requestor.GetExtra()) == 0 {
		return "", false
	}

	if len(requestor.GetExtra()[serviceaccount.NodeNameKey]) != 1 || requestor.GetExtra()[serviceaccount.NodeNameKey][0] != nodeName {
		return "", false
	}

	return nodeName, true
}
