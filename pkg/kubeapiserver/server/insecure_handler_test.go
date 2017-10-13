/*
Copyright 2017 The Kubernetes Authors.

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

package server

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server"
)

func setImpersonationHeaders(req *http.Request, u user.Info) {
	req.Header.Set(authenticationv1.ImpersonateUserHeader, u.GetName())
	for _, group := range u.GetGroups() {
		req.Header.Add(authenticationv1.ImpersonateGroupHeader, group)
	}
	for k, vv := range u.GetExtra() {
		for _, v := range vv {
			req.Header.Add(authenticationv1.ImpersonateUserExtraHeaderPrefix+k, v)
		}
	}
}

// normalizeUser converts a user.Info interface into a concrete user.DefaultInfo struct
// for comparison.
func normalizeUser(u user.Info) *user.DefaultInfo {
	return &user.DefaultInfo{
		Name:   u.GetName(),
		UID:    u.GetUID(),
		Groups: u.GetGroups(),
		Extra:  u.GetExtra(),
	}
}

func TestImpersonation(t *testing.T) {
	c := server.NewConfig(serializer.NewCodecFactory(runtime.NewScheme()))

	var lastUser user.Info
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ctx, ok := c.RequestContextMapper.Get(r); ok {
			if info, ok := request.UserFrom(ctx); ok {
				lastUser = info
			}
		}
		fmt.Fprint(w, "ok")
	})

	h := BuildInsecureHandlerChain(handler, c)

	tests := []struct {
		user *user.DefaultInfo
	}{
		{
			user: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			user: &user.DefaultInfo{
				Name:   "john",
				Groups: []string{"developers", "infra"},
			},
		},
		{
			user: &user.DefaultInfo{
				Name:   "john",
				Groups: []string{"developers", "infra"},
				Extra: map[string][]string{
					"scopes": []string{"read", "read:email"},
				},
			},
		},
	}

	for _, test := range tests {
		req := httptest.NewRequest("GET", "/", nil)
		req.Header.Set("Accept", "application/json")
		setImpersonationHeaders(req, test.user)

		lastUser = nil
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)

		if len(test.user.Groups) == 0 {
			// system:authenticated is only added if users don't request groups.
			test.user.Groups = append(test.user.Groups, user.AllAuthenticated)
		}

		got := normalizeUser(lastUser)
		want := normalizeUser(test.user)

		if !equality.Semantic.DeepEqual(want, got) {
			t.Errorf("wanted=%#v, got=%#v", want, got)
		}
	}
}

// whitelistAuthorizer allows all requests by the provided users.
type whitelistAuthorizer struct {
	users sets.String
}

func (w *whitelistAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	if w.users.Has(a.GetUser().GetName()) {
		return true, "", nil
	}
	return false, "user not in white list", nil
}

func TestAuthorization(t *testing.T) {
	c := server.NewConfig(serializer.NewCodecFactory(runtime.NewScheme()))
	c.Authorizer = &whitelistAuthorizer{
		users: sets.NewString("jane"),
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "ok")
	})
	h := BuildInsecureHandlerChain(handler, c)

	tests := []struct {
		user *user.DefaultInfo
		code int
	}{
		{
			// No impersionation requested. Request should be allowed.
			code: http.StatusOK,
		},
		{
			user: &user.DefaultInfo{Name: "jane"},
			code: http.StatusOK,
		},
		{
			user: &user.DefaultInfo{Name: "dave"},
			code: http.StatusForbidden,
		},
	}

	for _, test := range tests {
		req := httptest.NewRequest("GET", "/", nil)
		req.Header.Set("Accept", "application/json")
		if test.user != nil {
			setImpersonationHeaders(req, test.user)
		}

		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if test.code != rr.Code {
			t.Errorf("user %#v wanted=%d, got=%d", test.user, test.code, rr.Code)
			t.Log(rr.Body.String())
		}
	}
}
