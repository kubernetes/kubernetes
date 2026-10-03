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

package v1beta1

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	authorization "k8s.io/kubernetes/pkg/apis/authorization"
)

func TestConvert_authorization_SelfSubjectAccessReviewSpec_To_v1beta1_SelfSubjectAccessReviewSpec(t *testing.T) {
	unconditional := &authorization.AuthorizationOptions{
		HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
			authorization.ConditionsAwareDecisionTypeAllow,
			authorization.ConditionsAwareDecisionTypeDeny,
			authorization.ConditionsAwareDecisionTypeNoOpinion,
		},
	}
	conditional := &authorization.AuthorizationOptions{
		HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
			authorization.ConditionsAwareDecisionTypeAllow,
			authorization.ConditionsAwareDecisionTypeDeny,
			authorization.ConditionsAwareDecisionTypeNoOpinion,
			authorization.ConditionsAwareDecisionTypeConditionsMap,
			authorization.ConditionsAwareDecisionTypeUnion,
		},
	}
	resourceAttrs := &authorization.ResourceAttributes{
		Namespace: "ns", Verb: "get", Resource: "pods",
	}
	tests := []struct {
		name    string
		in      authorization.SelfSubjectAccessReviewSpec
		want    authorizationv1beta1.SelfSubjectAccessReviewSpec
		wantErr bool
	}{
		{
			name: "nil AuthorizationOptions converts",
			in: authorization.SelfSubjectAccessReviewSpec{
				ResourceAttributes: resourceAttrs,
			},
			want: authorizationv1beta1.SelfSubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1beta1.ResourceAttributes{
					Namespace: "ns", Verb: "get", Resource: "pods",
				},
			},
		},
		{
			name: "unconditional AuthorizationOptions accepted and dropped",
			in: authorization.SelfSubjectAccessReviewSpec{
				ResourceAttributes:   resourceAttrs,
				AuthorizationOptions: unconditional,
			},
			want: authorizationv1beta1.SelfSubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1beta1.ResourceAttributes{
					Namespace: "ns", Verb: "get", Resource: "pods",
				},
			},
		},
		{
			name: "conditional AuthorizationOptions rejected",
			in: authorization.SelfSubjectAccessReviewSpec{
				ResourceAttributes:   resourceAttrs,
				AuthorizationOptions: conditional,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got authorizationv1beta1.SelfSubjectAccessReviewSpec
			err := Convert_authorization_SelfSubjectAccessReviewSpec_To_v1beta1_SelfSubjectAccessReviewSpec(&tt.in, &got, nil)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected out (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvert_authorization_SubjectAccessReviewSpec_To_v1beta1_SubjectAccessReviewSpec(t *testing.T) {
	unconditional := &authorization.AuthorizationOptions{
		HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
			authorization.ConditionsAwareDecisionTypeAllow,
			authorization.ConditionsAwareDecisionTypeDeny,
			authorization.ConditionsAwareDecisionTypeNoOpinion,
		},
	}
	conditional := &authorization.AuthorizationOptions{
		HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
			authorization.ConditionsAwareDecisionTypeAllow,
			authorization.ConditionsAwareDecisionTypeDeny,
			authorization.ConditionsAwareDecisionTypeNoOpinion,
			authorization.ConditionsAwareDecisionTypeUnion,
		},
	}
	tests := []struct {
		name    string
		in      authorization.SubjectAccessReviewSpec
		want    authorizationv1beta1.SubjectAccessReviewSpec
		wantErr bool
	}{
		{
			name: "nil AuthorizationOptions propagates all other fields",
			in: authorization.SubjectAccessReviewSpec{
				User:   "alice",
				Groups: []string{"admins", "devs"},
				Extra:  map[string]authorization.ExtraValue{"scope": {"read", "write"}},
				UID:    "uid-1",
			},
			want: authorizationv1beta1.SubjectAccessReviewSpec{
				User:   "alice",
				Groups: []string{"admins", "devs"},
				Extra:  map[string]authorizationv1beta1.ExtraValue{"scope": {"read", "write"}},
				UID:    "uid-1",
			},
		},
		{
			name: "unconditional AuthorizationOptions accepted and dropped",
			in: authorization.SubjectAccessReviewSpec{
				User:                 "bob",
				AuthorizationOptions: unconditional,
			},
			want: authorizationv1beta1.SubjectAccessReviewSpec{
				User: "bob",
			},
		},
		{
			name: "conditional AuthorizationOptions rejected",
			in: authorization.SubjectAccessReviewSpec{
				User:                 "carol",
				AuthorizationOptions: conditional,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got authorizationv1beta1.SubjectAccessReviewSpec
			err := Convert_authorization_SubjectAccessReviewSpec_To_v1beta1_SubjectAccessReviewSpec(&tt.in, &got, nil)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected out (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(t *testing.T) {
	tests := []struct {
		name    string
		in      authorization.SubjectAccessReviewStatus
		want    authorizationv1beta1.SubjectAccessReviewStatus
		wantErr bool
	}{
		{
			name: "unconditional allow propagates",
			in: authorization.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "rbac: role/x allowed",
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "rbac: role/x allowed",
			},
		},
		{
			name: "unconditional deny propagates with EvaluationError",
			in: authorization.SubjectAccessReviewStatus{
				Denied:          true,
				Reason:          "webhook: denied",
				EvaluationError: "flaky evaluator",
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Denied:          true,
				Reason:          "webhook: denied",
				EvaluationError: "flaky evaluator",
			},
		},
		{
			name: "non-nil ConditionalDecision rejected",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got authorizationv1beta1.SubjectAccessReviewStatus
			err := Convert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(&tt.in, &got, nil)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected out (-want +got):\n%s", diff)
			}
		})
	}
}
