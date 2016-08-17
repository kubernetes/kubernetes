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

package apiserver

import (
	"net/http"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// WithImpersonation is a filter that will inspect and check requests that attempt to change the user.Info for their requests
func WithImpersonation(handler http.Handler, requestContextMapper api.RequestContextMapper, a authorizer.Authorizer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		requestedUser := req.Header.Get(authenticationapi.ImpersonateUserHeader)
		if len(requestedUser) == 0 {
			if len(req.Header[authenticationapi.ImpersonateGroupHeader]) > 0 {
				glog.V(4).Infof("attempt to impersonate groups without impersonating a user: %v", req.Header[authenticationapi.ImpersonateGroupHeader])
				forbidden(w, req)
				return
			}

			handler.ServeHTTP(w, req)
			return
		}

		impersonationRequests := buildImpersonationRequests(requestedUser, req.Header[authenticationapi.ImpersonateGroupHeader])

		ctx, exists := requestContextMapper.Get(req)
		if !exists {
			forbidden(w, req)
			return
		}
		requestor, exists := api.UserFrom(ctx)
		if !exists {
			forbidden(w, req)
			return
		}

		// if groups are not specified, then we need to look them up differently depending on the type of user
		// if they are specified, then they are the authority
		groupsSpecified := len(req.Header[authenticationapi.ImpersonateGroupHeader]) > 0

		// make sure we're allowed to impersonate each thing we're requesting.  While we're iterating through, start building username
		// and group information
		username := ""
		groups := []string{}
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

			default:
				glog.V(4).Infof("unknown impersonation request type: %v", impersonationRequest)
				forbidden(w, req)
				return
			}

			allowed, reason, err := a.Authorize(actingAsAttributes)
			if err != nil || !allowed {
				glog.V(4).Infof("Forbidden: %#v, Reason: %s, Error: %v", req.RequestURI, reason, err)
				forbidden(w, req)
				return
			}
		}

		newUser := &user.DefaultInfo{
			Name:   username,
			Groups: groups,
			Extra:  map[string][]string{},
		}
		requestContextMapper.Update(req, api.WithUser(ctx, newUser))

		oldUser, _ := api.UserFrom(ctx)
		httplog.LogOf(req, w).Addf("%v is acting as %v", oldUser, newUser)

		handler.ServeHTTP(w, req)
	})
}

// buildImpersonationRequests returns a list of objectreferences that represent the different things we're requesting to impersonate.
// Each request must be authorized against the current user before switching contexts
func buildImpersonationRequests(requestedUser string, requestedGroups []string) []api.ObjectReference {
	impersonationRequests := []api.ObjectReference{}

	if namespace, name, err := serviceaccount.SplitUsername(requestedUser); err == nil {
		impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "ServiceAccount", Namespace: namespace, Name: name})
	} else {
		impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "User", Name: requestedUser})
	}

	for _, group := range requestedGroups {
		impersonationRequests = append(impersonationRequests, api.ObjectReference{Kind: "Group", Name: group})
	}

	return impersonationRequests
}
