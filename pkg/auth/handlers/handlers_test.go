/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

func TestAuthenticateRequest(t *testing.T) {
	success := make(chan struct{})
	context := NewUserRequestContext()
	auth := NewRequestAuthenticator(
		context,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return &user.DefaultInfo{Name: "user"}, true, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Errorf("unexpected call to failed")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			if user, ok := context.Get(req); user == nil || !ok {
				t.Errorf("no user stored on context: %#v", context)
			}
			close(success)
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-success
	if len(context.requests) > 0 {
		t.Errorf("context should have no stored requests: %v", context)
	}
}

func TestAuthenticateRequestFailed(t *testing.T) {
	failed := make(chan struct{})
	context := NewUserRequestContext()
	auth := NewRequestAuthenticator(
		context,
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
	if len(context.requests) > 0 {
		t.Errorf("context should have no stored requests: %v", context)
	}
}

func TestAuthenticateRequestError(t *testing.T) {
	failed := make(chan struct{})
	context := NewUserRequestContext()
	auth := NewRequestAuthenticator(
		context,
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
	if len(context.requests) > 0 {
		t.Errorf("context should have no stored requests: %v", context)
	}
}
