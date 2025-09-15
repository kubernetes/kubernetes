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
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/warning"
)

func TestAuthenticateRequest(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		if token != "token" {
			t.Errorf("unexpected token: %s", token)
		}
		return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}}, true, nil
	}))
	resp, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if !ok || resp == nil || err != nil {
		t.Errorf("expected valid user")
	}
}

func TestAuthenticateRequestIncludingValueAfterToken(t *testing.T) {
	testCases := []struct {
		Req *http.Request
	}{
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"Bearer token a"}}}},
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"Bearer token a b c"}}}},
		{Req: &http.Request{Header: http.Header{"Authorization": []string{"Bearer token   a"}}}},
	}
	for i, testCase := range testCases {
		auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
			if token != "token" {
				t.Errorf("unexpected token: %s", token)
			}
			return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}}, true, nil
		}))
		resp, ok, err := auth.AuthenticateRequest(testCase.Req)
		if !ok || resp == nil || err != nil {
			t.Errorf("%d: expected valid user", i)
		}
	}
}

func TestAuthenticateRequestTokenInvalid(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		return nil, false, nil
	}))
	resp, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || resp != nil {
		t.Errorf("expected not authenticated user")
	}
	if err != invalidToken {
		t.Errorf("expected invalidToken error, got %v", err)
	}
}

func TestAuthenticateRequestTokenInvalidCustomError(t *testing.T) {
	customError := errors.New("custom")
	auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		return nil, false, customError
	}))
	resp, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || resp != nil {
		t.Errorf("expected not authenticated user")
	}
	if err != customError {
		t.Errorf("expected custom error, got %v", err)
	}
}

func TestAuthenticateRequestTokenError(t *testing.T) {
	auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		return nil, false, errors.New("error")
	}))
	resp, ok, err := auth.AuthenticateRequest(&http.Request{
		Header: http.Header{"Authorization": []string{"Bearer token"}},
	})
	if ok || resp != nil || err == nil {
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
		auth := New(authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
			t.Errorf("authentication should not have been called")
			return nil, false, nil
		}))
		user, ok, err := auth.AuthenticateRequest(testCase.Req)
		if ok || user != nil || err != nil {
			t.Errorf("%d: expected not authenticated (no token)", i)
		}
	}
}

type dummyRecorder struct {
	agent string
	text  string
}

func (r *dummyRecorder) AddWarning(agent, text string) {
	r.agent = agent
	r.text = text
	return
}

func (r *dummyRecorder) getWarning() string {
	return r.text
}

var _ warning.Recorder = &dummyRecorder{}

func TestBearerToken(t *testing.T) {
	tests := map[string]struct {
		AuthorizationHeaders []string
		TokenAuth            authenticator.Token

		ExpectedUserName             string
		ExpectedOK                   bool
		ExpectedErr                  bool
		ExpectedAuthorizationHeaders []string
		ExpectedRecordedWarning      string
	}{
		"no header": {
			AuthorizationHeaders:         nil,
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: nil,
		},
		"empty header": {
			AuthorizationHeaders:         []string{""},
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: []string{""},
		},
		"non-bearer header": {
			AuthorizationHeaders:         []string{"Basic 123"},
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: []string{"Basic 123"},
		},
		"empty bearer token": {
			AuthorizationHeaders:         []string{"Bearer "},
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: []string{"Bearer "},
		},
		"valid bearer token removing header": {
			AuthorizationHeaders: []string{"Bearer 123"},
			TokenAuth: authenticator.TokenFunc(func(ctx context.Context, t string) (*authenticator.Response, bool, error) {
				return &authenticator.Response{User: &user.DefaultInfo{Name: "myuser"}}, true, nil
			}),
			ExpectedUserName:             "myuser",
			ExpectedOK:                   true,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: nil,
		},
		"invalid bearer token": {
			AuthorizationHeaders:         []string{"Bearer 123"},
			TokenAuth:                    authenticator.TokenFunc(func(ctx context.Context, t string) (*authenticator.Response, bool, error) { return nil, false, nil }),
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  true,
			ExpectedAuthorizationHeaders: []string{"Bearer 123"},
		},
		"valid bearer token with a space": {
			AuthorizationHeaders:         []string{"Bearer  token"},
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  false,
			ExpectedAuthorizationHeaders: []string{"Bearer  token"},
			ExpectedRecordedWarning:      invalidTokenWithSpaceWarning,
		},
		"error bearer token": {
			AuthorizationHeaders: []string{"Bearer 123"},
			TokenAuth: authenticator.TokenFunc(func(ctx context.Context, t string) (*authenticator.Response, bool, error) {
				return nil, false, errors.New("error")
			}),
			ExpectedUserName:             "",
			ExpectedOK:                   false,
			ExpectedErr:                  true,
			ExpectedAuthorizationHeaders: []string{"Bearer 123"},
		},
	}

	for k, tc := range tests {
		dc := dummyRecorder{agent: "", text: ""}
		ctx := genericapirequest.NewDefaultContext()
		ctxWithRecorder := warning.WithWarningRecorder(ctx, &dc)
		req, _ := http.NewRequestWithContext(ctxWithRecorder, "GET", "/", nil)
		for _, h := range tc.AuthorizationHeaders {
			req.Header.Add("Authorization", h)
		}

		bearerAuth := New(tc.TokenAuth)
		resp, ok, err := bearerAuth.AuthenticateRequest(req)
		if tc.ExpectedErr != (err != nil) {
			t.Errorf("%s: Expected err=%v, got %v", k, tc.ExpectedErr, err)
			continue
		}
		if ok != tc.ExpectedOK {
			t.Errorf("%s: Expected ok=%v, got %v", k, tc.ExpectedOK, ok)
			continue
		}
		if ok && resp.User.GetName() != tc.ExpectedUserName {
			t.Errorf("%s: Expected username=%v, got %v", k, tc.ExpectedUserName, resp.User.GetName())
			continue
		}
		if len(tc.ExpectedRecordedWarning) > 0 && tc.ExpectedRecordedWarning != dc.getWarning() {
			t.Errorf("%s: Expected recorded warning=%v, got %v", k, tc.ExpectedRecordedWarning, dc.getWarning())
			continue
		}
		if !reflect.DeepEqual(req.Header["Authorization"], tc.ExpectedAuthorizationHeaders) {
			t.Errorf("%s: Expected headers=%#v, got %#v", k, tc.ExpectedAuthorizationHeaders, req.Header["Authorization"])
			continue
		}
	}
}
