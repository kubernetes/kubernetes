/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/auth/authorizer"
)

type mockAuthzHandler struct {
	isAuthorized bool
	err          error
}

func (mock *mockAuthzHandler) Authorize(a authorizer.Attributes) error {
	if mock.err != nil {
		return mock.err
	}
	if !mock.isAuthorized {
		return errors.New("Request unauthorized")
	} else {
		return nil
	}
}

func TestAuthorizationSecondPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{isAuthorized: false}
	handler2 := &mockAuthzHandler{isAuthorized: true}
	authzHandler := New(handler1, handler2)

	err := authzHandler.Authorize(nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestAuthorizationFirstPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{isAuthorized: true}
	handler2 := &mockAuthzHandler{isAuthorized: false}
	authzHandler := New(handler1, handler2)

	err := authzHandler.Authorize(nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestAuthorizationNonePasses(t *testing.T) {
	handler1 := &mockAuthzHandler{isAuthorized: false}
	handler2 := &mockAuthzHandler{isAuthorized: false}
	authzHandler := New(handler1, handler2)

	err := authzHandler.Authorize(nil)
	if err == nil {
		t.Errorf("Expected error: %v", err)
	}
}
