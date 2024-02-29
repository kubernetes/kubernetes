/*
Copyright 2023 The Kubernetes Authors.

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

package validating

import (
	"context"
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestCachingAuthorizer(t *testing.T) {
	type result struct {
		decision authorizer.Decision
		reason   string
		error    error
	}

	type invocation struct {
		attributes authorizer.Attributes
		expected   result
	}

	for _, tc := range []struct {
		name    string
		calls   []invocation
		backend []result
	}{
		{
			name: "hit",
			calls: []invocation{
				{
					attributes: authorizer.AttributesRecord{Name: "test name"},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
				{
					attributes: authorizer.AttributesRecord{Name: "test name"},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
			},
			backend: []result{
				{
					decision: authorizer.DecisionAllow,
					reason:   "test reason",
					error:    fmt.Errorf("test error"),
				},
			},
		},
		{
			name: "hit with differently-ordered groups",
			calls: []invocation{
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{
							Groups: []string{"a", "b", "c"},
						},
					},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{
							Groups: []string{"c", "b", "a"},
						},
					},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
			},
			backend: []result{
				{
					decision: authorizer.DecisionAllow,
					reason:   "test reason",
					error:    fmt.Errorf("test error"),
				},
			},
		},
		{
			name: "hit with differently-ordered extra",
			calls: []invocation{
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{
							Extra: map[string][]string{
								"k": {"a", "b", "c"},
							},
						},
					},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{
							Extra: map[string][]string{
								"k": {"c", "b", "a"},
							},
						},
					},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason",
						error:    fmt.Errorf("test error"),
					},
				},
			},
			backend: []result{
				{
					decision: authorizer.DecisionAllow,
					reason:   "test reason",
					error:    fmt.Errorf("test error"),
				},
			},
		},
		{
			name: "miss due to different name",
			calls: []invocation{
				{
					attributes: authorizer.AttributesRecord{Name: "alpha"},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason alpha",
						error:    fmt.Errorf("test error alpha"),
					},
				},
				{
					attributes: authorizer.AttributesRecord{Name: "beta"},
					expected: result{
						decision: authorizer.DecisionDeny,
						reason:   "test reason beta",
						error:    fmt.Errorf("test error beta"),
					},
				},
			},
			backend: []result{
				{
					decision: authorizer.DecisionAllow,
					reason:   "test reason alpha",
					error:    fmt.Errorf("test error alpha"),
				},
				{
					decision: authorizer.DecisionDeny,
					reason:   "test reason beta",
					error:    fmt.Errorf("test error beta"),
				},
			},
		},
		{
			name: "miss due to different user",
			calls: []invocation{
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{Name: "alpha"},
					},
					expected: result{
						decision: authorizer.DecisionAllow,
						reason:   "test reason alpha",
						error:    fmt.Errorf("test error alpha"),
					},
				},
				{
					attributes: authorizer.AttributesRecord{
						User: &user.DefaultInfo{Name: "beta"},
					},
					expected: result{
						decision: authorizer.DecisionDeny,
						reason:   "test reason beta",
						error:    fmt.Errorf("test error beta"),
					},
				},
			},
			backend: []result{
				{
					decision: authorizer.DecisionAllow,
					reason:   "test reason alpha",
					error:    fmt.Errorf("test error alpha"),
				},
				{
					decision: authorizer.DecisionDeny,
					reason:   "test reason beta",
					error:    fmt.Errorf("test error beta"),
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var misses int
			frontend := newCachingAuthorizer(func() authorizer.Authorizer {
				return authorizer.AuthorizerFunc(func(_ context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
					if misses >= len(tc.backend) {
						t.Fatalf("got more than expected %d backend invocations", len(tc.backend))
					}
					result := tc.backend[misses]
					misses++
					return result.decision, result.reason, result.error
				})
			}())

			for i, invocation := range tc.calls {
				decision, reason, err := frontend.Authorize(context.TODO(), invocation.attributes)
				if decision != invocation.expected.decision {
					t.Errorf("(call %d of %d) expected decision %v, got %v", i+1, len(tc.calls), invocation.expected.decision, decision)
				}
				if reason != invocation.expected.reason {
					t.Errorf("(call %d of %d) expected reason %q, got %q", i+1, len(tc.calls), invocation.expected.reason, reason)
				}
				if err.Error() != invocation.expected.error.Error() {
					t.Errorf("(call %d of %d) expected error %q, got %q", i+1, len(tc.calls), invocation.expected.error.Error(), err.Error())
				}
			}

			if len(tc.backend) > misses {
				t.Errorf("expected %d backend invocations, got %d", len(tc.backend), misses)
			}
		})
	}
}
