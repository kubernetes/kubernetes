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
	"reflect"
	"strings"
	"sync"
	"testing"
)

type constrainedImpersonateAuthorizer struct {
	checkedAttrs []authorizer.AttributesRecord
}

func (c *constrainedImpersonateAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	c.checkedAttrs = append(c.checkedAttrs, *copyAuthorizerAttr(a))

	user := a.GetUser()

	if user.GetName() == "sa-impersonater" && a.GetVerb() == "impersonate:serviceaccount" && a.GetResource() == "serviceaccounts" {
		return authorizer.DecisionAllow, "", nil
	}

	if user.GetName() == "system:serviceaccount:default:node" && a.GetVerb() == "impersonate:node" && a.GetResource() == "nodes" {
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

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "group-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" {
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
		expectedAttributes      []authorizer.AttributesRecord
		expectedUser            user.Info
		expectedCode            int
	}{
		{
			name: "impersonating-error",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "users",
					Name:       "anyone",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name: "tester",
					},
					ResourceRequest: true,
				},
				{
					Resource: "users",
					Name:     "anyone",
					Verb:     "impersonate",
					User: &user.DefaultInfo{
						Name: "tester",
					},
					ResourceRequest: true,
				},
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
			expectedUser: &user.DefaultInfo{
				Name:   "anyone",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "users",
					Name:       "anyone",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name: "user-impersonater",
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name: "user-impersonater",
					},
					ResourceRequest: true,
				},
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
			name: "impersonating-sa-allowed",
			user: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			expectedUser: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:default",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:default", "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "serviceaccounts",
					Name:       "default",
					Namespace:  "default",
					Verb:       "impersonate:serviceaccount",
					User: &user.DefaultInfo{
						Name: "sa-impersonater",
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name: "sa-impersonater",
					},
					ResourceRequest: true,
				},
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
			expectedUser: &user.DefaultInfo{
				Name: "sa-impersonater",
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:node",
					User: &user.DefaultInfo{
						Name: "sa-impersonater",
					},
					ResourceRequest: true,
				},
				{
					Resource: "groups",
					Verb:     "impersonate",
					Name:     "system:nodes",
					User: &user.DefaultInfo{
						Name: "sa-impersonater",
					},
					ResourceRequest: true,
				},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
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
				Name: "node-impersonater",
			},
			expectedUser: &user.DefaultInfo{
				Name: "node-impersonater",
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:node",
					User: &user.DefaultInfo{
						Name: "node-impersonater",
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:create",
					User: &user.DefaultInfo{
						Name: "node-impersonater",
					},
					ResourceRequest: true,
				},
				{
					Resource: "groups",
					Verb:     "impersonate",
					Name:     "system:nodes",
					User: &user.DefaultInfo{
						Name: "node-impersonater",
					},
					ResourceRequest: true,
				},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
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
			expectedUser: &user.DefaultInfo{
				Name:   "system:node:node1",
				Groups: []string{user.NodesGroup, "system:authenticated"},
				Extra:  map[string][]string{},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:node",
					User: &user.DefaultInfo{
						Name: "node-impersonater",
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name: "node-impersonater",
					},
					ResourceRequest: true,
				},
			},
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
				Name:   "user-impersonater",
				Groups: []string{"group-impersonater"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "user-impersonater",
				Groups: []string{"group-impersonater"},
			},
			impersonationUser:       "system:admin",
			impersonationGroups:     []string{"extra-setter-scopes"},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a", "scope-b"}},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "users",
					Name:       "system:admin",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"group-impersonater"},
					},
					ResourceRequest: true,
				},
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "groups",
					Name:       "extra-setter-scopes",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"group-impersonater"},
					},
					ResourceRequest: true,
				},
				{
					APIGroup:    authenticationapi.SchemeGroupVersion.Group,
					APIVersion:  authenticationapi.SchemeGroupVersion.Version,
					Resource:    "userextras",
					Subresource: "scopes",
					Name:        "scope-a",
					Verb:        "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"group-impersonater"},
					},
					ResourceRequest: true,
				},
				{
					Resource: "users",
					Verb:     "impersonate",
					Name:     "system:admin",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"group-impersonater"},
					},
					ResourceRequest: true,
				},
			},
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
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{"scopes": {"scope-a", "scope-b"}},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "users",
					Name:       "system:admin",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"extra-setter-scopes"},
					},
					ResourceRequest: true,
				},
				{
					APIGroup:    authenticationapi.SchemeGroupVersion.Group,
					APIVersion:  authenticationapi.SchemeGroupVersion.Version,
					Resource:    "userextras",
					Subresource: "scopes",
					Name:        "scope-a",
					Verb:        "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"extra-setter-scopes"},
					},
					ResourceRequest: true,
				},
				{
					APIGroup:    authenticationapi.SchemeGroupVersion.Group,
					APIVersion:  authenticationapi.SchemeGroupVersion.Version,
					Resource:    "userextras",
					Subresource: "scopes",
					Name:        "scope-b",
					Verb:        "impersonate:user-info",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"extra-setter-scopes"},
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"extra-setter-scopes"},
					},
					ResourceRequest: true,
				},
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
			expectedUser: &user.DefaultInfo{
				Name:   "system:node:node1",
				Groups: []string{user.NodesGroup, "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:scheduled-node",
					User: &user.DefaultInfo{
						Name:   "system:serviceaccount:default:default",
						Groups: []string{"scheduled-node-impersonater"},
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name:   "system:serviceaccount:default:default",
						Groups: []string{"scheduled-node-impersonater"},
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
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
			expectedUser: &user.DefaultInfo{
				Name:   "user-impersonater",
				Groups: []string{"scheduled-node-impersonater"},
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"node1"},
				},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:node",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"scheduled-node-impersonater"},
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
				{
					Resource: "groups",
					Verb:     "impersonate",
					Name:     "system:nodes",
					User: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"scheduled-node-impersonater"},
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
			requestInfo: &request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				APIVersion:        "v1",
				Resource:          "pods",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-node-if-sa-can-impersonate-node",
			user: &user.DefaultInfo{
				Name: "system:serviceaccount:default:node",
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"node1"},
				},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "system:node:node1",
				Groups: []string{user.NodesGroup, "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:scheduled-node",
					User: &user.DefaultInfo{
						Name: "system:serviceaccount:default:node",
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "nodes",
					Name:       "node1",
					Verb:       "impersonate:node",
					User: &user.DefaultInfo{
						Name: "system:serviceaccount:default:node",
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
				{
					APIVersion: "v1",
					Resource:   "pods",
					Verb:       "impersonate-on:get",
					User: &user.DefaultInfo{
						Name: "system:serviceaccount:default:node",
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					ResourceRequest: true,
				},
			},
			impersonationUser:   "system:node:node1",
			impersonationGroups: []string{user.NodesGroup},
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
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedAttributes: []authorizer.AttributesRecord{
				{
					APIGroup:   authenticationapi.SchemeGroupVersion.Group,
					APIVersion: authenticationapi.SchemeGroupVersion.Version,
					Resource:   "users",
					Name:       "system:admin",
					Verb:       "impersonate:user-info",
					User: &user.DefaultInfo{
						Name: "legacy-impersonater",
					},
					ResourceRequest: true,
				},
				{
					Resource: "users",
					Verb:     "impersonate",
					Name:     "system:admin",
					User: &user.DefaultInfo{
						Name: "legacy-impersonater",
					},
					ResourceRequest: true,
				},
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
	var actualUser user.Info
	var lock sync.Mutex

	doNothingHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		currentCtx := req.Context()
		user, exists := request.UserFrom(currentCtx)
		if !exists {
			actualUser = nil
			return
		}

		actualUser = user

		if _, ok := req.Header[authenticationapi.ImpersonateUserHeader]; ok {
			t.Fatal("user header still present")
		}
		if _, ok := req.Header[authenticationapi.ImpersonateGroupHeader]; ok {
			t.Fatal("group header still present")
		}
		for key := range req.Header {
			if strings.HasPrefix(key, authenticationapi.ImpersonateUserExtraHeaderPrefix) {
				t.Fatalf("extra header still present: %v", key)
			}
		}
		if _, ok := req.Header[authenticationapi.ImpersonateUIDHeader]; ok {
			t.Fatal("uid header still present")
		}
	})

	constrainedAuthorizer := &constrainedImpersonateAuthorizer{}
	handler := func(delegate http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			/*defer func() {
				if r := recover(); r != nil {
					t.Errorf("Recovered %v", r)
				}
			}()*/

			lock.Lock()
			defer lock.Unlock()
			req = req.WithContext(ctx)
			currentCtx := req.Context()

			user, exists := request.UserFrom(currentCtx)
			if !exists {
				actualUser = nil
				return
			} else {
				actualUser = user
			}

			delegate.ServeHTTP(w, req)
		})
	}(WithConstrainedImpersonation(doNothingHandler, constrainedAuthorizer, serializer.NewCodecFactory(runtime.NewScheme())))
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

			if !reflect.DeepEqual(actualUser, tc.expectedUser) {
				t.Errorf("%s: expected %#v, actual %#v", tc.name, tc.expectedUser, actualUser)
				return
			}

			if !reflect.DeepEqual(constrainedAuthorizer.checkedAttrs, tc.expectedAttributes) {
				t.Errorf("%s: expected %#v, actual %#v", tc.name, len(tc.expectedAttributes), len(constrainedAuthorizer.checkedAttrs))
			}
			// clean the attrs after each test case.
			constrainedAuthorizer.checkedAttrs = []authorizer.AttributesRecord{}
		})
	}
}
