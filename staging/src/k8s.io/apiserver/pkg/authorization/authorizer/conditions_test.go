/*
Copyright 2026 The Kubernetes Authors.

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

package authorizer_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestConditionsAwareDecision(t *testing.T) {
	unexpectedErr := fmt.Errorf("unexpected things happened")
	otherErr := fmt.Errorf("other error")

	ctx := t.Context()
	sampleAttrs := authorizer.AttributesRecord{}
	samplePref := authorizer.ConditionsEncodingPreference{}

	tests := []struct {
		name            string
		testDecisions   []authorizer.ConditionsAwareDecision
		wantIsAllowed   bool
		wantIsNoOpinion bool
		wantIsDenied    bool
		wantReason      string
		wantAnyError    bool
		wantErrorIs     error
		wantString      string
	}{
		{
			name: "zero value",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecision{},
				authorizer.ConditionsAwareDecisionFromParts(0, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (named1 authorizer.Decision, named2 string, named3 error) {
					return
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied: true,
			wantReason:   "",
			wantErrorIs:  nil,
			wantString:   `Deny("", <nil>)`,
		},
		{
			name: "deny constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionDeny("foo", unexpectedErr),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionDeny, "foo", unexpectedErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "foo", unexpectedErr
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied: true,
			wantReason:   "foo",
			wantErrorIs:  unexpectedErr,
			wantString:   `Deny("foo", "unexpected things happened")`,
		},
		{
			name: "allow constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionAllow("ok", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionAllow, "ok", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "ok", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsAllowed: true,
			wantReason:    "ok",
			wantErrorIs:   nil,
			wantString:    `Allow("ok", <nil>)`,
		},
		{
			name: "noopinion constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionNoOpinion, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionNoOpinion, "", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsNoOpinion: true,
			wantReason:      "",
			wantErrorIs:     nil,
			wantString:      `NoOpinion("", <nil>)`,
		},
		{
			name: "from parts: unsupported mode",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied: true,
			wantReason:   "",
			wantAnyError: true,
			wantString:   `Deny("", "unknown unconditional decision type: 42")`,
		},
		{
			name: "from parts: unsupported mode with other error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "foo", otherErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "foo", otherErr
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied: true,
			wantReason:   "foo",
			wantErrorIs:  otherErr,
			wantString:   `Deny("foo", "[other error, unknown unconditional decision type: 42]")`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, d := range tt.testDecisions {
				t.Run(fmt.Sprint(i), func(t *testing.T) {
					isAllowed := d.IsAllowed()
					if isAllowed != tt.wantIsAllowed {
						t.Errorf("IsAllowed() = %v, want %v", isAllowed, tt.wantIsAllowed)
					}
					isNoOpinion := d.IsNoOpinion()
					if isNoOpinion != tt.wantIsNoOpinion {
						t.Errorf("IsNoOpinion() = %v, want %v", isNoOpinion, tt.wantIsNoOpinion)
					}
					isDenied := d.IsDenied()
					if isDenied != tt.wantIsDenied {
						t.Errorf("IsDenied() = %v, want %v", isDenied, tt.wantIsDenied)
					}
					gotReason := d.Reason()
					if gotReason != tt.wantReason {
						t.Errorf("Reason() = %v, want %v", gotReason, tt.wantReason)
					}
					gotError := d.Error()
					if tt.wantAnyError {
						if gotError == nil {
							t.Errorf("Error() = %v, want some error", nil)
						}
					} else {
						if !errors.Is(gotError, tt.wantErrorIs) {
							t.Errorf("Error() = %v, want %v", gotError, tt.wantErrorIs)
						}
					}

					gotString := d.String()
					if gotString != tt.wantString {
						t.Errorf("String() = %v, want %v", gotString, tt.wantString)
					}
				})
			}
		})
	}
}
