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
	authenticationapi "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

type constrainedImpersonateAuthorizer struct{}

func (constrainedImpersonateAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	user := a.GetUser()

	if user.GetName() == "sa-impersonater" && a.GetVerb() == "impersonate:serviceaccount" && a.GetResource() == "serviceaccounts" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() == "node-impersonater" && a.GetVerb() == "impersonate:node" && a.GetResource() == "nodes" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "scheduled-node-impersonater" && a.GetVerb() == "impersonate:scheduled-node" && a.GetResource() == "nodes" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() == "user-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() == "group-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "extra-setter-scopes" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "userextras" && a.GetSubresource() == "scopes" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() == "legacy-impersonater" && a.GetVerb() == "impersonate" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() != "legacy-impersonator" && (a.GetVerb() == "impersonate-on:list" || a.GetVerb() == "impersonate-on:get") {
		return authorizer.DecisionAllow, "", nil
	}

	return authorizer.DecisionNoOpinion, "deny by default", nil
}

func TestConstrainedImpersonationFilter(t *testing.T) {
	testCases := []struct {
		name                    string
		user                    user.Info
		impersonationUser       string
		impersonationGroups     []string
		impersonationUserExtras map[string][]string
		impersonationUid        string
		requestInfo             *request.RequestInfo
		expectedCode            int
	}{
		{
			name: "impersonating-error",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			impersonationUser: "anyone",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "impersonating-user-allowed",
			user: &user.DefaultInfo{
				Name: "user-impersonater",
			},
			impersonationUser: "anyone",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "sa-impersonator-impersonating-user-not-allowed",
			user: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			impersonationUser: "anyone",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "impersonating-sa-allowed",
			user: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			impersonationUser: "system:serviceaccount:default:default",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "impersonating-node-not-allowed",
			user: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "impersonating-node-not-allowed-action",
			user: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "impersonating-node-allowed",
			user: &user.DefaultInfo{
				Name: "node-impersonater",
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "disallowed-userextra-3",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "extra-setter-particular-scopes"},
			},
			impersonationUser:       "system:admin",
			impersonationGroups:     []string{"some-group"},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a", "scope-b"}},
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-userextras",
			user: &user.DefaultInfo{
				Name:   "user-impersonater",
				Groups: []string{"extra-setter-scopes"},
			},
			impersonationUser:       "system:admin",
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a", "scope-b"}},
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "allowed-scheduled-node",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:default",
				Groups: []string{"scheduled-node-impersonater"},
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"node1"},
				},
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "disallowed-scheduled-node-without-sa",
			user: &user.DefaultInfo{
				Name:   "user-impersonater",
				Groups: []string{"scheduled-node-impersonater"},
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"node1"},
				},
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-scheduled-node-without-node-impersonator",
			user: &user.DefaultInfo{
				Name:   "node-impersonater",
				Groups: []string{"scheduled-node-impersonater"},
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"node1"},
				},
			},
			impersonationUser: "system:node:node1",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "allowed-legacy-impersonator",
			user: &user.DefaultInfo{
				Name: "legacy-impersonater",
			},
			impersonationUser: "system:admin",
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusOK,
		},
	}

	var ctx context.Context
	var lock sync.Mutex

	doNothingHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {})
	handler := func(delegate http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			lock.Lock()
			defer lock.Unlock()
			req = req.WithContext(ctx)

			delegate.ServeHTTP(w, req)
		})
	}(WithContrainedImpersonation(doNothingHandler, constrainedImpersonateAuthorizer{}, serializer.NewCodecFactory(runtime.NewScheme())))
	server := httptest.NewServer(handler)
	defer server.Close()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx = request.WithUser(request.NewContext(), tc.user)
			ctx = request.WithRequestInfo(ctx, tc.requestInfo)

			req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}
			if len(tc.impersonationUser) > 0 {
				req.Header.Add(authenticationapi.ImpersonateUserHeader, tc.impersonationUser)
			}
			for _, group := range tc.impersonationGroups {
				req.Header.Add(authenticationapi.ImpersonateGroupHeader, group)
			}
			for extraKey, values := range tc.impersonationUserExtras {
				for _, value := range values {
					req.Header.Add(authenticationapi.ImpersonateUserExtraHeaderPrefix+extraKey, value)
				}
			}
			if len(tc.impersonationUid) > 0 {
				req.Header.Add(authenticationapi.ImpersonateUIDHeader, tc.impersonationUid)
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}
			if resp.StatusCode != tc.expectedCode {
				t.Errorf("%s: expected %v, actual %v", tc.name, tc.expectedCode, resp.StatusCode)
				return
			}
		})
	}
}
