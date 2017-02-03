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
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/client-go/pkg/api"
	authenticationapi "k8s.io/client-go/pkg/apis/authentication"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// WithImpersonation is a filter that will inspect and check requests that attempt to change the user.Info for their requests
func WithImpersonation(handler http.Handler, requestContextMapper request.RequestContextMapper, a authorizer.Authorizer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		impersonationRequests, err := buildImpersonationRequests(req.Header)
		if err != nil {
			glog.V(4).Infof("%v", err)
			responsewriters.InternalError(w, req, err)
			return
		}
		if len(impersonationRequests) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		ctx, exists := requestContextMapper.Get(req)
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no context found for request"))
			return
		}
		requestor, exists := request.UserFrom(ctx)
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		// if groups are not specified, then we need to look them up differently depending on the type of user
		// if they are specified, then they are the authority
		groupsSpecified := len(req.Header[authenticationapi.ImpersonateGroupHeader]) > 0

		// make sure we're allowed to impersonate each thing we're requesting.  While we're iterating through, start building username
		// and group information
		username := ""
		groups := []string{}
		userExtra := map[string][]string{}
		for _, impersonationRequest := range impersonationRequests {
			actingAsAttributes := &authorizer.AttributesRecord{
				User:            requestor,
				Verb:            "impersonate",
				APIGroup:        impersonationRequest.GetObjectKind().GroupVersionKind().Group,
				Namespace:       impersonationRequest.Namespace,
				Name:            impersonationRequest.Name,
				ResourceRequest: true,
			}

			switch impersonationRequest.GetObjectKind().GroupVersionKind().GroupKind() {
			case api.Kind("ServiceAccount"):
				actingAsAttributes.Resource = "serviceaccounts"
				username = serviceaccount.MakeUsername(impersonationRequest.Namespace, impersonationRequest.Name)
				if !groupsSpecified {
					// if groups aren't specified for a service account, we know the groups because its a fixed mapping.  Add them
					groups = serviceaccount.MakeGroupNames(impersonationRequest.Namespace, impersonationRequest.Name)
				}

			case api.Kind("User"):
				actingAsAttributes.Resource = "users"
				username = impersonationRequest.Name

			case api.Kind("Group"):
				actingAsAttributes.Resource = "groups"
				groups = append(groups, impersonationRequest.Name)

			case authenticationapi.Kind("UserExtra"):
				extraKey := impersonationRequest.FieldPath
				extraValue := impersonationRequest.Name
				actingAsAttributes.Resource = "userextras"
				actingAsAttributes.Subresource = extraKey
				userExtra[extraKey] = append(userExtra[extraKey], extraValue)

			default:
				glog.V(4).Infof("unknown impersonation request type: %v", impersonationRequest)
				responsewriters.Forbidden(actingAsAttributes, w, req, fmt.Sprintf("unknown impersonation request type: %v", impersonationRequest))
				return
			}

			allowed, reason, err := a.Authorize(actingAsAttributes)
			if err != nil || !allowed {
				glog.V(4).Infof("Forbidden: %#v, Reason: %s, Error: %v", req.RequestURI, reason, err)
				responsewriters.Forbidden(actingAsAttributes, w, req, reason)
				return
			}
		}

		newUser := &user.DefaultInfo{
			Name:   username,
			Groups: groups,
			Extra:  userExtra,
		}
		requestContextMapper.Update(req, request.WithUser(ctx, newUser))

		oldUser, _ := request.UserFrom(ctx)
		httplog.LogOf(req, w).Addf("%v is acting as %v", oldUser, newUser)

		// clear all the impersonation headers from the request
		req.Header.Del(authenticationapi.ImpersonateUserHeader)
		req.Header.Del(authenticationapi.ImpersonateGroupHeader)
		for headerName := range req.Header {
			if strings.HasPrefix(headerName, authenticationapi.ImpersonateUserExtraHeaderPrefix) {
				req.Header.Del(headerName)
			}
		}

		handler.ServeHTTP(w, req)
	})
}

// buildImpersonationRequests returns a list of objectreferences that represent the different things we're requesting to impersonate.
// Also includes a map[string][]string representing user.Info.Extra
// Each request must be authorized against the current user before switching contexts.
func buildImpersonationRequests(headers http.Header) ([]api.ObjectReference, error) {
	impersonationRequests := []api.ObjectReference{}

	requestedUser := headers.Get(authenticationapi.ImpersonateUserHeader)
	hasUser := len(requestedUser) > 0
	if hasUser {
		if namespace, name, err := serviceaccount.SplitUsername(requestedUser); err == nil {
			impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "ServiceAccount", Namespace: namespace, Name: name})
		} else {
			impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "User", Name: requestedUser})
		}
	}

	hasGroups := false
	for _, group := range headers[authenticationapi.ImpersonateGroupHeader] {
		hasGroups = true
		impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "Group", Name: group})
	}

	hasUserExtra := false
	for headerName, values := range headers {
		if !strings.HasPrefix(headerName, authenticationapi.ImpersonateUserExtraHeaderPrefix) {
			continue
		}

		hasUserExtra = true
		extraKey := strings.ToLower(headerName[len(authenticationapi.ImpersonateUserExtraHeaderPrefix):])

		// make a separate request for each extra value they're trying to set
		for _, value := range values {
			impersonationRequests = append(impersonationRequests,
				api.ObjectReference{
					Kind: "UserExtra",
					// we only parse out a group above, but the parsing will fail if there isn't SOME version
					// using the internal version will help us fail if anyone starts using it
					APIVersion: authenticationapi.SchemeGroupVersion.String(),
					Name:       value,
					// ObjectReference doesn't have a subresource field.  FieldPath is close and available, so we'll use that
					// TODO fight the good fight for ObjectReference to refer to resources and subresources
					FieldPath: extraKey,
				})
		}
	}

	if (hasGroups || hasUserExtra) && !hasUser {
		return nil, fmt.Errorf("requested %v without impersonating a user", impersonationRequests)
	}

	return impersonationRequests, nil
}
