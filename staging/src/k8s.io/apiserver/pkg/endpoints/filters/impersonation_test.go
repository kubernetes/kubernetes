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
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	authenticationapi "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/runtime"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
)

type impersonateAuthorizer struct{}

func (impersonateAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	user := a.GetUser()

	switch {
	case user.GetName() == "system:admin":
		return authorizer.DecisionAllow, "", nil

	case user.GetName() == "tester":
		return authorizer.DecisionNoOpinion, "", fmt.Errorf("works on my machine")

	case user.GetName() == "deny-me":
		return authorizer.DecisionNoOpinion, "denied", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "wheel" && a.GetVerb() == "impersonate" && a.GetResource() == "users" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "sa-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "serviceaccounts" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "regular-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "users" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 1 && user.GetGroups()[1] == "group-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "groups" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 1 && user.GetGroups()[1] == "extra-setter-scopes" && a.GetVerb() == "impersonate" && a.GetResource() == "userextras" && a.GetSubresource() == "scopes" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 1 && (user.GetGroups()[1] == "escaped-scopes" || user.GetGroups()[1] == "almost-escaped-scopes") {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 1 && user.GetGroups()[1] == "extra-setter-particular-scopes" &&
		a.GetVerb() == "impersonate" && a.GetResource() == "userextras" && a.GetSubresource() == "scopes" && a.GetName() == "scope-a" && a.GetAPIGroup() == "authentication.k8s.io" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 1 && user.GetGroups()[1] == "extra-setter-project" && a.GetVerb() == "impersonate" && a.GetResource() == "userextras" && a.GetSubresource() == "project" && a.GetAPIGroup() == "authentication.k8s.io" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "everything-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "users" && a.GetAPIGroup() == "" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "everything-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "uids" && a.GetName() == "some-uid" && a.GetAPIGroup() == "authentication.k8s.io" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "everything-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "groups" && a.GetAPIGroup() == "" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(user.GetGroups()) > 0 && user.GetGroups()[0] == "everything-impersonater" && a.GetVerb() == "impersonate" && a.GetResource() == "userextras" && a.GetSubresource() == "scopes" && a.GetAPIGroup() == "authentication.k8s.io" {
		return authorizer.DecisionAllow, "", nil
	}

	return authorizer.DecisionNoOpinion, "deny by default", nil
}

func TestImpersonationFilter(t *testing.T) {
	testCases := []struct {
		name                    string
		user                    user.Info
		impersonationUser       string
		impersonationGroups     []string
		impersonationUserExtras map[string][]string
		impersonationUid        string
		expectedUser            user.Info
		expectedCode            int
	}{
		{
			name: "not-impersonating",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "impersonating-error",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			impersonationUser: "anyone",
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "impersonating-group-without-user",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			impersonationGroups: []string{"some-group"},
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedCode: http.StatusInternalServerError,
		},
		{
			name: "impersonating-extra-without-user",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a"}},
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedCode: http.StatusInternalServerError,
		},
		{
			name: "impersonating-uid-without-user",
			user: &user.DefaultInfo{
				Name: "tester",
			},
			impersonationUid: "some-uid",
			expectedUser: &user.DefaultInfo{
				Name: "tester",
			},
			expectedCode: http.StatusInternalServerError,
		},
		{
			name: "disallowed-group",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel"},
			},
			impersonationUser:   "system:admin",
			impersonationGroups: []string{"some-group"},
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel"},
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-group",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "group-impersonater"},
			},
			impersonationUser:   "system:admin",
			impersonationGroups: []string{"some-group"},
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"some-group", "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "disallowed-userextra-1",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel"},
			},
			impersonationUser:       "system:admin",
			impersonationGroups:     []string{"some-group"},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a"}},
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel"},
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "disallowed-userextra-2",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "extra-setter-project"},
			},
			impersonationUser:       "system:admin",
			impersonationGroups:     []string{"some-group"},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a"}},
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "extra-setter-project"},
			},
			expectedCode: http.StatusForbidden,
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
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "extra-setter-particular-scopes"},
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-userextras",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "extra-setter-scopes"},
			},
			impersonationUser:       "system:admin",
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a", "scope-b"}},
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{"scopes": {"scope-a", "scope-b"}},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "percent-escaped-userextras",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "escaped-scopes"},
			},
			impersonationUser:       "system:admin",
			impersonationUserExtras: map[string][]string{"example.com%2fescaped%e1%9b%84scopes": {"scope-a", "scope-b"}},
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{"example.com/escapedᛄscopes": {"scope-a", "scope-b"}},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "almost-percent-escaped-userextras",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "almost-escaped-scopes"},
			},
			impersonationUser:       "system:admin",
			impersonationUserExtras: map[string][]string{"almost%zzpercent%xxencoded": {"scope-a", "scope-b"}},
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{"almost%zzpercent%xxencoded": {"scope-a", "scope-b"}},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "allowed-users-impersonation",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"regular-impersonater"},
			},
			impersonationUser: "tester",
			expectedUser: &user.DefaultInfo{
				Name:   "tester",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "disallowed-impersonating",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"sa-impersonater"},
			},
			impersonationUser: "tester",
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"sa-impersonater"},
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-sa-impersonating",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"sa-impersonater"},
				Extra:  map[string][]string{},
			},
			impersonationUser: "system:serviceaccount:foo:default",
			expectedUser: &user.DefaultInfo{
				Name:   "system:serviceaccount:foo:default",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:foo", "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "anonymous-username-prevents-adding-authenticated-group",
			user: &user.DefaultInfo{
				Name: "system:admin",
			},
			impersonationUser: "system:anonymous",
			expectedUser: &user.DefaultInfo{
				Name:   "system:anonymous",
				Groups: []string{"system:unauthenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "unauthenticated-group-prevents-adding-authenticated-group",
			user: &user.DefaultInfo{
				Name: "system:admin",
			},
			impersonationUser:   "unknown",
			impersonationGroups: []string{"system:unauthenticated"},
			expectedUser: &user.DefaultInfo{
				Name:   "unknown",
				Groups: []string{"system:unauthenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "unauthenticated-group-prevents-double-adding-authenticated-group",
			user: &user.DefaultInfo{
				Name: "system:admin",
			},
			impersonationUser:   "unknown",
			impersonationGroups: []string{"system:authenticated"},
			expectedUser: &user.DefaultInfo{
				Name:   "unknown",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "specified-authenticated-group-prevents-double-adding-authenticated-group",
			user: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"wheel", "group-impersonater"},
			},
			impersonationUser:   "system:admin",
			impersonationGroups: []string{"some-group", "system:authenticated"},
			expectedUser: &user.DefaultInfo{
				Name:   "system:admin",
				Groups: []string{"some-group", "system:authenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "anonymous-user-should-include-unauthenticated-group",
			user: &user.DefaultInfo{
				Name: "system:admin",
			},
			impersonationUser: "system:anonymous",
			expectedUser: &user.DefaultInfo{
				Name:   "system:anonymous",
				Groups: []string{"system:unauthenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "anonymous-user-prevents-double-adding-unauthenticated-group",
			user: &user.DefaultInfo{
				Name: "system:admin",
			},
			impersonationUser:   "system:anonymous",
			impersonationGroups: []string{"system:unauthenticated"},
			expectedUser: &user.DefaultInfo{
				Name:   "system:anonymous",
				Groups: []string{"system:unauthenticated"},
				Extra:  map[string][]string{},
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "allowed-user-impersonation-with-uid",
			user: &user.DefaultInfo{
				Name: "dev",
				Groups: []string{
					"everything-impersonater",
				},
			},
			impersonationUser: "tester",
			impersonationUid:  "some-uid",
			expectedUser: &user.DefaultInfo{
				Name:   "tester",
				Groups: []string{"system:authenticated"},
				Extra:  map[string][]string{},
				UID:    "some-uid",
			},
			expectedCode: http.StatusOK,
		},
		{
			name: "disallowed-user-impersonation-with-uid",
			user: &user.DefaultInfo{
				Name: "dev",
				Groups: []string{
					"everything-impersonater",
				},
			},
			impersonationUser: "tester",
			impersonationUid:  "disallowed-uid",
			expectedUser: &user.DefaultInfo{
				Name:   "dev",
				Groups: []string{"everything-impersonater"},
			},
			expectedCode: http.StatusForbidden,
		},
		{
			name: "allowed-impersonation-with-all-headers",
			user: &user.DefaultInfo{
				Name: "dev",
				Groups: []string{
					"everything-impersonater",
				},
			},
			impersonationUser:       "tester",
			impersonationUid:        "some-uid",
			impersonationGroups:     []string{"system:authenticated"},
			impersonationUserExtras: map[string][]string{"scopes": {"scope-a", "scope-b"}},
			expectedUser: &user.DefaultInfo{
				Name:   "tester",
				Groups: []string{"system:authenticated"},
				UID:    "some-uid",
				Extra:  map[string][]string{"scopes": {"scope-a", "scope-b"}},
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
	handler := func(delegate http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Recovered %v", r)
				}
			}()
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
	}(WithImpersonation(doNothingHandler, impersonateAuthorizer{}, serializer.NewCodecFactory(runtime.NewScheme())))

	server := httptest.NewServer(handler)
	defer server.Close()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			func() {
				lock.Lock()
				defer lock.Unlock()
				ctx = request.WithUser(request.NewContext(), tc.user)
			}()

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
		})
	}
}
