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

package rest

import (
	"context"
	"errors"
	"reflect"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// mockAuthorizer provides a mock implementation of the authorizer.Interface.
type mockAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (a *mockAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return a.decision, a.reason, a.err
}

func TestEnsureAuthorizedForVerb(t *testing.T) {
	tests := []struct {
		name          string
		ctx           context.Context
		authorizer    authorizer.Authorizer
		verb          string
		expectErr     string
		expectErrType interface{}
	}{
		{
			name:          "no request info in context",
			ctx:           context.Background(),
			verb:          "create",
			expectErr:     `Internal error occurred: no request info in context`,
			expectErrType: &apierrors.StatusError{},
		},
		{
			name: "verb already matches",
			ctx: genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
				Verb: "create",
			}),
			verb: "create",
		},
		{
			name: "nil authorizer",
			ctx: genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
				Verb: "get",
			}),
			authorizer:    nil,
			verb:          "create",
			expectErr:     `Internal error occurred: no authorizer available`,
			expectErrType: &apierrors.StatusError{},
		},
		{
			name:       "authorizer returns error",
			ctx:        genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{Verb: "get"}),
			authorizer: &mockAuthorizer{err: errors.New("auth error")},
			verb:       "create",
			expectErr:  "auth error",
		},
		{
			name:          "authorizer denies",
			ctx:           genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{Verb: "get", Resource: "pods", Name: "my-pod"}),
			authorizer:    &mockAuthorizer{decision: authorizer.DecisionDeny, reason: "no reason"},
			verb:          "create",
			expectErr:     `pods "my-pod" is forbidden: no reason`,
			expectErrType: &apierrors.StatusError{},
		},
		{
			name:       "authorizer allows",
			ctx:        genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{Verb: "get"}),
			authorizer: &mockAuthorizer{decision: authorizer.DecisionAllow},
			verb:       "create",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ensureAuthorizedForVerb(tc.ctx, tc.authorizer, tc.verb)

			if len(tc.expectErr) > 0 {
				if err == nil {
					t.Fatalf("expected error %q, got nil", tc.expectErr)
				}
				if err.Error() != tc.expectErr {
					t.Errorf("expected error %q, got %q", tc.expectErr, err.Error())
				}
				if tc.expectErrType != nil {
					if reflect.TypeOf(err) != reflect.TypeOf(tc.expectErrType) {
						t.Errorf("expected error type %T, got %T", tc.expectErrType, err)
					}
				}
			} else if err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
		})
	}
}
