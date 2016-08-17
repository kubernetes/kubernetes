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

package handlers

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

func TestAuthenticateRequest(t *testing.T) {
	success := make(chan struct{})
	contextMapper := api.NewRequestContextMapper()
	auth, err := NewRequestAuthenticator(
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return &user.DefaultInfo{Name: "user"}, true, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Errorf("unexpected call to failed")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			ctx, ok := contextMapper.Get(req)
			if ctx == nil || !ok {
				t.Errorf("no context stored on contextMapper: %#v", contextMapper)
			}
			user, ok := api.UserFrom(ctx)
			if user == nil || !ok {
				t.Errorf("no user stored in context: %#v", ctx)
			}
			close(success)
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-success
	empty, err := api.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

func TestAuthenticateRequestFailed(t *testing.T) {
	failed := make(chan struct{})
	contextMapper := api.NewRequestContextMapper()
	auth, err := NewRequestAuthenticator(
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return nil, false, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
	empty, err := api.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

func TestAuthenticateRequestError(t *testing.T) {
	failed := make(chan struct{})
	contextMapper := api.NewRequestContextMapper()
	auth, err := NewRequestAuthenticator(
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return nil, false, errors.New("failure")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
	empty, err := api.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

type mockAuthenticator struct {
	user user.Info
}

func (m *mockAuthenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	if m.user == nil {
		return nil, false, nil
	}
	return m.user, true, nil
}

func TestAuthenticationInfoHeader(t *testing.T) {
	a := new(mockAuthenticator)

	auth, err := NewRequestAuthenticator(
		api.NewRequestContextMapper(),
		a,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			t.Error("unexpected call to failed handler")
			w.Write([]byte(`{}`))
		}),

		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(`{}`))
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		user       user.Info
		wantHeader string
	}{
		{
			user:       &user.DefaultInfo{},
			wantHeader: `username="", uid=""`,
		},
		{
			user: &user.DefaultInfo{
				Name: "jane",
				UID:  "42",
			},
			wantHeader: `username="jane", uid="42"`,
		},
		{
			user: &user.DefaultInfo{
				Name: `foo"bar`, // Ensure values are properly escaped
				UID:  "42",
			},
			wantHeader: `username="foo\"bar", uid="42"`,
		},
		{
			user: &user.DefaultInfo{
				Name: "Schrödinger",
				UID:  "",
			},
			wantHeader: `username="Schr\u00f6dinger", uid=""`,
		},
		{
			user: &user.DefaultInfo{
				Name: "\n",
				UID:  "",
			},
			wantHeader: `username="\n", uid=""`,
		},
	}

	for _, test := range tests {
		a.user = test.user

		rr := httptest.NewRecorder()
		auth.ServeHTTP(rr, &http.Request{})

		authInfo := rr.Header().Get("Authentication-Info")
		if test.wantHeader != authInfo {
			t.Errorf("expected Authentication-Info=%q, got=%q", test.wantHeader, authInfo)
		}
	}
}
