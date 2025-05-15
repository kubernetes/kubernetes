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

package cel

import (
	"context"
	"testing"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCELMatcherEval(t *testing.T) {
	tests := []struct {
		name                string
		celExpression       string
		subjectAccessReview *authorizationv1.SubjectAccessReview
		expectedMatch       bool
		expectError         bool
	}{
		{
			name:          "matching user condition",
			celExpression: "request.user == 'test-user'",
			subjectAccessReview: &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					User: "test-user",
				},
			},
			expectedMatch: true,
			expectError:   false,
		},
		{
			name:          "non-matching user condition",
			celExpression: "request.user == 'test-user'",
			subjectAccessReview: &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					User: "other-user",
				},
			},
			expectedMatch: false,
			expectError:   false,
		},
		{
			name:          "matching extra fields condition",
			celExpression: "'admin' in request.extra.roles",
			subjectAccessReview: &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					Extra: map[string]authorizationv1.ExtraValue{
						"roles": {"admin", "user"},
					},
				},
			},
			expectedMatch: true,
			expectError:   false,
		},
		{
			name:          "compound condition with resource attributes",
			celExpression: "request.user == 'admin' && request.resourceAttributes.namespace == 'default' && request.resourceAttributes.verb == 'update'",
			subjectAccessReview: &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					User: "admin",
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Namespace: "kube-system",
						Verb:      "update",
						Resource:  "pods",
					},
				},
			},
			expectedMatch: false,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			compiler := NewDefaultCompiler()
			expression := &SubjectAccessReviewMatchCondition{
				Expression: tt.celExpression,
			}

			compilationResult, err := compiler.CompileCELExpression(expression)
			if err != nil {
				t.Fatalf("failed to compile expression: %v", err)
			}

			matcher := CELMatcher{
				CompilationResults: []CompilationResult{compilationResult},
			}

			subjectAccessReview := tt.subjectAccessReview
			subjectAccessReview.TypeMeta = metav1.TypeMeta{
				APIVersion: authorizationv1.SchemeGroupVersion.String(),
				Kind:       "SubjectAccessReview",
			}
			match, err := matcher.Eval(context.Background(), tt.subjectAccessReview)

			if tt.expectError && err == nil {
				t.Errorf("expected error but got none")
			}

			if !tt.expectError && err != nil {
				t.Errorf("expected no error but got: %v", err)
			}

			if match != tt.expectedMatch {
				t.Errorf("expected match result %v but got %v", tt.expectedMatch, match)
			}
		})
	}
}
