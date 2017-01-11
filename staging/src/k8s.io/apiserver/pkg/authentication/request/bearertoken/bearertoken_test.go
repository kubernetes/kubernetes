/*
Copyright 2014 The Kubernetes Authors.

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

package bearertoken

import (
	"errors"
	"net/http"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestAuthenticateRequest(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		if token != "token" {
			t.Errorf("unexpected token: %s", token)
		}
		return &user.DefaultInfo{Name: "user"}, true, nil
	}))
	user, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if !ok || user == nil || err != nil {
		t.Errorf("expected valid user")
	}
}

func TestAuthenticateRequestTokenInvalid(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		return nil, false, nil
	}))
	user, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || user != nil {
		t.Errorf("expected not authenticated user")
	}
	if err != invalidToken {
		t.Errorf("expected invalidToken error, got %v", err)
	}
}

func TestAuthenticateRequestTokenInvalidCustomError(t *testing.T) {
	customError := errors.New("custom")
	auth := New(authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		return nil, false, customError
	}))
	user, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || user != nil {
		t.Errorf("expected not authenticated user")
	}
	if err != customError {
		t.Errorf("expected custom error, got %v", err)
	}
}

func TestAuthenticateRequestTokenError(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		return nil, false, errors.New("error")
	}))
	user, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || user != nil || err == nil {
		t.Errorf("expected error")
	}
}

func TestAuthenticateRequestBadValue(t *testing.T) {
	testCases := []struct {
		Req *http.Request
	}{
		{Req: &http.Request{}},
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"Bearer"}}}},
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"bear token"}}}},
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"Bearer: token"}}}},
	}
	for i, testCase := range testCases {
		auth := New(authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
			t.Errorf("authentication should not have been called")
			return nil, false, nil
		}))
		user, ok, err := auth.AuthenticateRequest(testCase.Req)
		if ok || user != nil || err != nil {
			t.Errorf("%d: expected not authenticated (no token)", i)
		}
	}
}
