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

package union

import (
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/auth/user"
)

type mockAuthRequestHandler struct {
	returnUser      user.Info
	isAuthenticated bool
	err             error
}

var (
	user1 = &user.DefaultInfo{Name: "fresh_ferret", UID: "alfa"}
	user2 = &user.DefaultInfo{Name: "elegant_sheep", UID: "bravo"}
)

func (mock *mockAuthRequestHandler) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	return mock.returnUser, mock.isAuthenticated, mock.err
}

func TestAuthenticateRequestSecondPasses(t *testing.T) {
	handler1 := &mockAuthRequestHandler{returnUser: user1}
	handler2 := &mockAuthRequestHandler{returnUser: user2, isAuthenticated: true}
	authRequestHandler := New(handler1, handler2)
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	authenticatedUser, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !isAuthenticated {
		t.Errorf("Unexpectedly unauthenticated: %v", isAuthenticated)
	}
	if !reflect.DeepEqual(user2, authenticatedUser) {
		t.Errorf("Expected %v, got %v", user2, authenticatedUser)
	}
}

func TestAuthenticateRequestFirstPasses(t *testing.T) {
	handler1 := &mockAuthRequestHandler{returnUser: user1, isAuthenticated: true}
	handler2 := &mockAuthRequestHandler{returnUser: user2}
	authRequestHandler := New(handler1, handler2)
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	authenticatedUser, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !isAuthenticated {
		t.Errorf("Unexpectedly unauthenticated: %v", isAuthenticated)
	}
	if !reflect.DeepEqual(user1, authenticatedUser) {
		t.Errorf("Expected %v, got %v", user1, authenticatedUser)
	}
}

func TestAuthenticateRequestSuppressUnnecessaryErrors(t *testing.T) {
	handler1 := &mockAuthRequestHandler{err: errors.New("first")}
	handler2 := &mockAuthRequestHandler{isAuthenticated: true}
	authRequestHandler := New(handler1, handler2)
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	_, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !isAuthenticated {
		t.Errorf("Unexpectedly unauthenticated: %v", isAuthenticated)
	}
}

func TestAuthenticateRequestNoAuthenticators(t *testing.T) {
	authRequestHandler := New()
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	authenticatedUser, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if isAuthenticated {
		t.Errorf("Unexpectedly authenticated: %v", isAuthenticated)
	}
	if authenticatedUser != nil {
		t.Errorf("Unexpected authenticatedUser: %v", authenticatedUser)
	}
}

func TestAuthenticateRequestNonePass(t *testing.T) {
	handler1 := &mockAuthRequestHandler{}
	handler2 := &mockAuthRequestHandler{}
	authRequestHandler := New(handler1, handler2)
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	_, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if isAuthenticated {
		t.Errorf("Unexpectedly authenticated: %v", isAuthenticated)
	}
}

func TestAuthenticateRequestAdditiveErrors(t *testing.T) {
	handler1 := &mockAuthRequestHandler{err: errors.New("first")}
	handler2 := &mockAuthRequestHandler{err: errors.New("second")}
	authRequestHandler := New(handler1, handler2)
	req, _ := http.NewRequest("GET", "http://example.org", nil)

	_, isAuthenticated, err := authRequestHandler.AuthenticateRequest(req)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if !strings.Contains(err.Error(), "first") {
		t.Errorf("Expected error containing %v, got %v", "first", err)
	}
	if !strings.Contains(err.Error(), "second") {
		t.Errorf("Expected error containing %v, got %v", "second", err)
	}
	if isAuthenticated {
		t.Errorf("Unexpectedly authenticated: %v", isAuthenticated)
	}
}
