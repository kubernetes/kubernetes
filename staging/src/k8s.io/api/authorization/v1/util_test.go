/*
Copyright The Kubernetes Authors.

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

package v1

// NOTE: This file MUST be kept in sync with pkg/apis/authorization/util_test.go
import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestAuthorizationOptions_Supports(t *testing.T) {
	tests := []struct {
		name              string
		ao                *AuthorizationOptions
		wantConditional   bool
		wantUnconditional bool
	}{
		{
			name:              "nil AuthorizationOptions falls back to unconditional",
			ao:                nil,
			wantConditional:   false,
			wantUnconditional: true,
		},
		{
			name:              "empty HandledDecisionTypes supports neither",
			ao:                &AuthorizationOptions{},
			wantConditional:   false,
			wantUnconditional: false,
		},
		{
			name: "only unconditional types",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeDeny,
					ConditionsAwareDecisionTypeNoOpinion,
				},
			},
			wantConditional:   false,
			wantUnconditional: true,
		},
		{
			name: "all five types support both",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeDeny,
					ConditionsAwareDecisionTypeNoOpinion,
					ConditionsAwareDecisionTypeConditionsMap,
					ConditionsAwareDecisionTypeUnion,
				},
			},
			wantConditional:   true,
			wantUnconditional: true,
		},
		{
			name: "missing Union rejects conditional but keeps unconditional",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeDeny,
					ConditionsAwareDecisionTypeNoOpinion,
					ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			wantConditional:   false,
			wantUnconditional: true,
		},
		{
			name: "missing NoOpinion rejects both",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeDeny,
					ConditionsAwareDecisionTypeConditionsMap,
					ConditionsAwareDecisionTypeUnion,
				},
			},
			wantConditional:   false,
			wantUnconditional: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.ao.SupportsConditionalAuthorization(); got != tt.wantConditional {
				t.Errorf("SupportsConditionalAuthorization() = %v, want %v", got, tt.wantConditional)
			}
			if got := tt.ao.SupportsUnconditionalAuthorization(); got != tt.wantUnconditional {
				t.Errorf("SupportsUnconditionalAuthorization() = %v, want %v", got, tt.wantUnconditional)
			}
		})
	}
}

func TestAuthorizationOptions_GetHandledDecisionTypes(t *testing.T) {
	tests := []struct {
		name string
		ao   *AuthorizationOptions
		want sets.Set[ConditionsAwareDecisionType]
	}{
		{
			name: "nil returns unconditional set",
			ao:   nil,
			want: sets.New(
				ConditionsAwareDecisionTypeAllow,
				ConditionsAwareDecisionTypeDeny,
				ConditionsAwareDecisionTypeNoOpinion,
			),
		},
		{
			name: "empty returns empty set",
			ao:   &AuthorizationOptions{},
			want: sets.New[ConditionsAwareDecisionType](),
		},
		{
			name: "populated returns the exact set",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeUnion,
				},
			},
			want: sets.New(
				ConditionsAwareDecisionTypeAllow,
				ConditionsAwareDecisionTypeUnion,
			),
		},
		{
			name: "duplicates collapse",
			ao: &AuthorizationOptions{
				HandledDecisionTypes: []ConditionsAwareDecisionType{
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeAllow,
					ConditionsAwareDecisionTypeDeny,
				},
			},
			want: sets.New(
				ConditionsAwareDecisionTypeAllow,
				ConditionsAwareDecisionTypeDeny,
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.ao.GetHandledDecisionTypes()
			if !got.Equal(tt.want) {
				t.Errorf("GetHandledDecisionTypes() = %v, want %v", got.UnsortedList(), tt.want.UnsortedList())
			}
		})
	}
}

func TestDecisionTypeAccessorsReturnFreshCopies(t *testing.T) {
	// Mutating the returned set must not affect the package-level constants
	// or subsequent calls.
	tests := []struct {
		name string
		fn   func() sets.Set[ConditionsAwareDecisionType]
		want sets.Set[ConditionsAwareDecisionType]
	}{
		{
			name: "ConditionalAuthorizationDecisionTypes",
			fn:   ConditionalAuthorizationDecisionTypes,
			want: sets.New(
				ConditionsAwareDecisionTypeAllow,
				ConditionsAwareDecisionTypeDeny,
				ConditionsAwareDecisionTypeNoOpinion,
				ConditionsAwareDecisionTypeConditionsMap,
				ConditionsAwareDecisionTypeUnion,
			),
		},
		{
			name: "UnconditionalAuthorizationDecisionTypes",
			fn:   UnconditionalAuthorizationDecisionTypes,
			want: sets.New(
				ConditionsAwareDecisionTypeAllow,
				ConditionsAwareDecisionTypeDeny,
				ConditionsAwareDecisionTypeNoOpinion,
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			first := tt.fn()
			if !first.Equal(tt.want) {
				t.Fatalf("first call = %v, want %v", first.UnsortedList(), tt.want.UnsortedList())
			}
			first.Insert(ConditionsAwareDecisionType("bogus"))
			second := tt.fn()
			if !second.Equal(tt.want) {
				t.Errorf("second call = %v, want %v (returned set was not a fresh copy)", second.UnsortedList(), tt.want.UnsortedList())
			}
		})
	}
}
