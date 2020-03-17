package scopeauthorizer

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	kauthorizer "k8s.io/apiserver/pkg/authorization/authorizer"

	authorizationv1 "github.com/openshift/api/authorization/v1"
)

func TestAuthorize(t *testing.T) {
	testCases := []struct {
		name            string
		attributes      kauthorizer.AttributesRecord
		expectedAllowed kauthorizer.Decision
		expectedErr     string
		expectedMsg     string
	}{
		{
			name: "no user",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			expectedErr:     `user missing from context`,
		},
		{
			name: "no extra",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "empty extra",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{}},
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "empty scopes",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {}}},
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "bad scope",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"does-not-exist"}}},
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedMsg:     `scopes [does-not-exist] prevent this action, additionally the following non-fatal errors were reported: no scope evaluator found for "does-not-exist"`,
		},
		{
			name: "bad scope 2",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"role:dne"}}},
				ResourceRequest: true,
				Namespace:       "ns",
			},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedMsg:     `scopes [role:dne] prevent this action, additionally the following non-fatal errors were reported: bad format for scope role:dne`,
		},
		{
			name: "scope doesn't cover",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"user:info"}}},
				ResourceRequest: true,
				Namespace:       "ns",
				Verb:            "get", Resource: "users", Name: "harold"},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedMsg:     `scopes [user:info] prevent this action`,
		},
		{
			name: "scope covers",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"user:info"}}},
				ResourceRequest: true,
				Namespace:       "ns",
				Verb:            "get", Resource: "users", Name: "~"},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "scope covers for discovery",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"user:info"}}},
				ResourceRequest: false,
				Namespace:       "ns",
				Verb:            "get", Path: "/api"},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "user:full covers any resource",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"user:full"}}},
				ResourceRequest: true,
				Namespace:       "ns",
				Verb:            "update", Resource: "users", Name: "harold"},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
		{
			name: "user:full covers any non-resource",
			attributes: kauthorizer.AttributesRecord{
				User:            &user.DefaultInfo{Extra: map[string][]string{authorizationv1.ScopesKey: {"user:full"}}},
				ResourceRequest: false,
				Namespace:       "ns",
				Verb:            "post", Path: "/foo/bar/baz"},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			authorizer := NewAuthorizer(nil)

			actualAllowed, actualMsg, actualErr := authorizer.Authorize(context.TODO(), tc.attributes)
			switch {
			case len(tc.expectedErr) == 0 && actualErr == nil:
			case len(tc.expectedErr) == 0 && actualErr != nil:
				t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
			case len(tc.expectedErr) != 0 && actualErr == nil:
				t.Errorf("%s: missing error: %v", tc.name, tc.expectedErr)
			case len(tc.expectedErr) != 0 && actualErr != nil:
				if !strings.Contains(actualErr.Error(), tc.expectedErr) {
					t.Errorf("expected %v, got %v", tc.expectedErr, actualErr)
				}
			}
			if tc.expectedMsg != actualMsg {
				t.Errorf("expected %v, got %v", tc.expectedMsg, actualMsg)
			}
			if tc.expectedAllowed != actualAllowed {
				t.Errorf("expected %v, got %v", tc.expectedAllowed, actualAllowed)
			}
		})
	}
}
