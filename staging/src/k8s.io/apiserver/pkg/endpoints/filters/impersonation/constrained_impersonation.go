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

package impersonation

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	authenticationv1 "k8s.io/api/authentication/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/klog/v2"
)

// WithConstrainedImpersonation implements constrained impersonation as described in https://kep.k8s.io/5284
// It also includes a complete reimplementation of legacy impersonation for backwards compatibility.
// At a high level, constrained impersonation uses multiple authorization checks to allow for the granular
// expression of impersonation access.  For example, a service account may be authorized to impersonate the
// node that it is associated with but only when listing pods.  See the linked KEP for further details.
func WithConstrainedImpersonation(handler http.Handler, a authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	return &constrainedImpersonationHandler{
		handler: handler,
		tracker: newImpersonationModesTracker(a),
		s:       s,
	}
}

type constrainedImpersonationHandler struct {
	handler http.Handler
	tracker *impersonationModesTracker
	s       runtime.NegotiatedSerializer
}

func (c *constrainedImpersonationHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ctx := req.Context()

	wantedUser, err := processImpersonationHeaders(req.Header)
	if err != nil {
		responsewriters.RespondWithError(w, req, err, c.s)
		return
	}
	if wantedUser == nil { // impersonation was not attempted so skip to the next handler
		c.handler.ServeHTTP(w, req)
		return
	}
	attributes, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		responsewriters.InternalError(w, req, err)
		return
	}
	requestor := attributes.GetUser()
	if requestor == nil {
		responsewriters.InternalError(w, req, errors.New("no User found in the context"))
		return
	}

	impersonatedUser, err := c.tracker.getImpersonatedUser(ctx, wantedUser, attributes)
	if err != nil {
		klog.V(4).InfoS("Forbidden", "URI", req.RequestURI, "err", err)
		responsewriters.RespondWithError(w, req, err, c.s)
		return
	}

	req = req.WithContext(request.WithUser(ctx, impersonatedUser.user))
	httplog.LogOf(req, w).Addf("%v is impersonating %v", userString(requestor), userString(impersonatedUser.user))
	audit.LogImpersonatedUser(ctx, impersonatedUser.user, impersonatedUser.constraint)

	c.handler.ServeHTTP(w, req)
}

// processImpersonationHeaders converts the impersonation headers in the given input headers
// into the equivalent user.DefaultInfo.  The resulting user is a raw representation of the
// input headers, that is, no defaulting or other mutations have been applied to it.
func processImpersonationHeaders(headers http.Header) (*user.DefaultInfo, error) {
	wantedUser := &user.DefaultInfo{}

	wantedUser.Name = headers.Get(authenticationv1.ImpersonateUserHeader)
	hasUser := len(wantedUser.Name) > 0

	wantedUser.UID = headers.Get(authenticationv1.ImpersonateUIDHeader)
	hasUID := len(wantedUser.UID) > 0

	hasGroups := false
	for _, group := range headers[authenticationv1.ImpersonateGroupHeader] {
		hasGroups = true
		wantedUser.Groups = append(wantedUser.Groups, group)
	}

	hasUserExtra := false
	for headerName, values := range headers {
		if !strings.HasPrefix(headerName, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
			continue
		}

		hasUserExtra = true

		if len(values) == 0 {
			// this looks a little strange but matches the behavior of buildImpersonationRequests from legacy impersonation
			// http1 uses textproto.Reader#ReadMIMEHeader which does seem to allow an empty slice for values
			// http2 uses http.Header#Add which will cause the values slice to always be non-empty
			continue
		}

		extraKey := unescapeExtraKey(strings.ToLower(headerName[len(authenticationv1.ImpersonateUserExtraHeaderPrefix):]))

		if wantedUser.Extra == nil {
			wantedUser.Extra = map[string][]string{}
		}
		wantedUser.Extra[extraKey] = append(wantedUser.Extra[extraKey], values...)
	}

	if !hasUser && (hasUID || hasGroups || hasUserExtra) {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("requested %#v without impersonating a user name", wantedUser))
	}

	if !hasUser {
		return nil, nil
	}

	// clear all the impersonation headers from the request to prevent downstream layers from knowing that impersonation was used
	// we do not want anything outside of this package trying to behave differently based on if impersonation was used
	headers.Del(authenticationv1.ImpersonateUserHeader)
	headers.Del(authenticationv1.ImpersonateUIDHeader)
	headers.Del(authenticationv1.ImpersonateGroupHeader)
	for headerName := range headers {
		if strings.HasPrefix(headerName, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
			headers.Del(headerName)
		}
	}

	return wantedUser, nil
}

// impersonationModesTracker records which impersonation mode was last successful for a given requestor user.
// this allows us to check for the more secure constrained impersonation modes first while keeping the overall
// cost of legacy impersonation unchanged (as we will support legacy impersonation forever).
type impersonationModesTracker struct {
	modes    []impersonationMode
	idxCache *modeIndexCache
}

func newImpersonationModesTracker(a authorizer.Authorizer) *impersonationModesTracker {
	loggingAuthorizer := authorizer.AuthorizerFunc(func(ctx context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
		decision, reason, err := a.Authorize(ctx, attributes)
		// build a detailed log of the authorization
		// make the whole block conditional so we do not do a lot of string-building we will not use
		if klogV := klog.V(5); klogV.Enabled() { // same log level that the RBAC authorizer uses for verbose logging
			u := attributes.GetUser()
			fieldSelector, _ := attributes.GetFieldSelector()
			labelSelector, _ := attributes.GetLabelSelector()
			klogV.InfoSDepth(3, "Impersonation authorization check",
				// we cannot just pass attributes to the logger as that will not capture the actual result of calling these methods
				// impersonation makes heavy use of wrapping these methods to add extra logic
				"username", u.GetName(),
				"uid", u.GetUID(),
				"groups", u.GetGroups(),
				"extra", u.GetExtra(),

				"isResourceRequest", attributes.IsResourceRequest(),

				"namespace", attributes.GetNamespace(),
				"verb", attributes.GetVerb(),
				"group", attributes.GetAPIGroup(),
				"version", attributes.GetAPIVersion(),
				"resource", attributes.GetResource(),
				"subresource", attributes.GetSubresource(),
				"name", attributes.GetName(),
				"fieldSelector", fieldSelector,
				"labelSelector", labelSelector,

				"path", attributes.GetPath(),

				"decision", decision,
				"reason", reason,
				"err", err,
			)
		}
		return decision, reason, err
	})
	return &impersonationModesTracker{
		modes:    allImpersonationModes(loggingAuthorizer),
		idxCache: newModeIndexCache(),
	}
}

func (t *impersonationModesTracker) getImpersonatedUser(ctx context.Context, wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (*impersonatedUserInfo, error) {
	// share a single cache key across all modes so that we only lazily build it once
	key := &impersonationCacheKey{wantedUser: wantedUser, attributes: attributes}
	var firstErr error

	// try the last successful mode first to reduce the amortized cost of impersonation
	// we attempt all modes unless we short-circuit due to a successful impersonation
	modeIdx, modeIdxOk := t.idxCache.get(attributes)
	if modeIdxOk {
		impersonatedUser, err := t.modes[modeIdx].check(ctx, key, wantedUser, attributes)
		if err == nil && impersonatedUser != nil {
			return impersonatedUser, nil
		}
		firstErr = err
	}

	for i, mode := range t.modes {
		if modeIdxOk && i == modeIdx {
			continue // skip already attempted mode
		}

		impersonatedUser, err := mode.check(ctx, key, wantedUser, attributes)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if impersonatedUser == nil {
			continue
		}
		t.idxCache.set(attributes, i)
		return impersonatedUser, nil
	}

	if firstErr != nil {
		return nil, firstErr
	}

	// this should not happen, but make sure we fail closed when no impersonation mode succeeded
	return nil, errors.New("all impersonation modes failed")
}
