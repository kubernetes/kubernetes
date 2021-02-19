/*
Copyright 2020 The Kubernetes Authors.

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

package gkeautopilot

import (
	"context"
	"fmt"
	"testing"

	apiv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	clientv1 "k8s.io/client-go/listers/admissionregistration/v1"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/authentication/user"
)

type mockUserInfo struct {
	user.Info
	name   string
	groups []string
}

var (
	errMock = fmt.Errorf("some err")
)

func (u *mockUserInfo) GetName() string     { return u.name }
func (u *mockUserInfo) GetGroups() []string { return u.groups }

type mockRequest struct {
	authorizer.Attributes

	apiGroup    string
	apiVersion  string
	namespace   string
	resource    string
	name        string
	subresource string
	user        mockUserInfo
	verb        string

	isResourceRequest bool
}

func (r *mockRequest) GetAPIGroup() string     { return r.apiGroup }
func (r *mockRequest) GetAPIVersion() string   { return r.apiVersion }
func (r *mockRequest) GetNamespace() string    { return r.namespace }
func (r *mockRequest) GetResource() string     { return r.resource }
func (r *mockRequest) GetName() string         { return r.name }
func (r *mockRequest) GetSubresource() string  { return r.subresource }
func (r *mockRequest) GetUser() user.Info      { return &r.user }
func (r *mockRequest) GetVerb() string         { return r.verb }
func (r *mockRequest) IsResourceRequest() bool { return r.isResourceRequest }

type mockWebhookLister struct {
	clientv1.ValidatingWebhookConfigurationLister
	mockGetFn func(name string) (*apiv1.ValidatingWebhookConfiguration, error)
}

func (m *mockWebhookLister) Get(name string) (*apiv1.ValidatingWebhookConfiguration, error) {
	return m.mockGetFn(name)
}

func TestAuthorize(t *testing.T) {
	baseConfig := config{
		ManagedResources: []ManagedResources{
			{
				Resources: []ManagedResource{
					{
						//"/apis/group.foo.bar/v2/gadgets/pillar-obj"
						APIGroup:  "group.foo.bar",
						Namespace: "",
						Resource:  "gadgets",
						Name:      "pillar-obj",
					},
					{
						//"/api/v1/namespaces/kube-system/pods/coolpod",
						APIGroup:  "",
						Namespace: "kube-system",
						Resource:  "pods",
						Name:      "coolpod",
					},
					{
						//"/api/v1/namespaces/kube-system/pods/coolpod",
						APIGroup:  "",
						Namespace: "kube-system",
						Resource:  "pods",
						Name:      "coolpod",
						Subresources: []Subresource{
							{
								Name:         "exec",
								AllowedVerbs: []string{},
							},
							{
								Name:         "scale",
								AllowedVerbs: []string{"edit"},
							},
						},
					},
				},
				AllowedVerbs: []string{"get", "lowRiskVerb"},
			},
		},
	}

	type input struct {
		selected bool
		// indicates whether the request is not a resource request
		notResourceRequest        bool
		policyEnforcementDisabled bool
		prepConfig                func(*config)
		prepRequest               func()
	}

	type expected struct {
		decision authorizer.Decision
		reason   string
		errFunc  func() error
	}

	var request *mockRequest
	// if set, only the selected cases will run. This is used only for
	// debugging, and should otherwise be false
	runSelectedOnly := false
	testCases := map[string]struct {
		input    input
		expected expected
	}{
		"IgnoredUser_NoOpinion": {
			input{
				policyEnforcementDisabled: true,
				prepConfig: func(c *config) {
					c.IgnoredIdentities.Users = []string{"ignoredUser"}
				},

				prepRequest: func() {
					request.verb = impersonationVerb
					request.user = mockUserInfo{
						name: "ignoredUser",
					}
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"IgnoredGroup_NoOpinion": {
			input{
				policyEnforcementDisabled: true,
				prepConfig: func(c *config) {
					c.IgnoredIdentities.Groups = []string{"ignoredGroup"}
				},

				prepRequest: func() {
					request.verb = impersonationVerb
					request.user = mockUserInfo{
						groups: []string{"userGroup", "ignoredGroup", "anotherGroup"},
					}
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"Impersonating_IsDenied": {
			input{
				prepRequest: func() {
					request.verb = impersonationVerb
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   authReasonDeniedImpersonation,
			},
		},
		"NotResourceRequest_NoOpinion": {
			input{
				notResourceRequest: true,
			},

			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"PolicyEnforcementNotEnabled_IsDenied": {
			input{
				policyEnforcementDisabled: true,
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   authReasonDeniedPolicyEnforcementNotEnabled,
			},
		},
		"VerbDeniedForNamespace_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name:        "ns2",
							DeniedVerbs: []string{"verb1", "verb2", "verb3"},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb3"
					request.namespace = "ns2"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbManagedNamespace, "ns2", "verb3"),
			},
		},
		"VerbDeniedForNamespace_ResourceIgnored_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name:        "ns2",
							DeniedVerbs: []string{"verb1", "verb2", "verb3"},
							IgnoredResources: []ResourceSubresource{
								{
									Resource: "roles",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb3"
					request.namespace = "ns2"
					request.resource = "roles"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"VerbDeniedForNamespace_ResourceSubresourceIgnored_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name:        "ns2",
							DeniedVerbs: []string{"verb1", "verb2", "verb3"},
							IgnoredResources: []ResourceSubresource{
								{
									Resource: "roles",
									Subresource: "subres",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb3"
					request.namespace = "ns2"
					request.resource = "roles"
					request.subresource = "subres"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"VerbDeniedForNamespace_ResourceSubresourceIgnored_SubresourceNotMatch_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name:        "ns2",
							DeniedVerbs: []string{"verb1", "verb2", "verb3"},
							IgnoredResources: []ResourceSubresource{
								{
									Resource: "roles",
									Subresource: "subres1",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb3"
					request.namespace = "ns2"
					request.resource = "roles"
					request.subresource = "subres2"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbManagedNamespace, "ns2", "verb3"),
			},
		},
		"ResourceDeniedForNamespaceAndSubresourceNonEmpty_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "ns2",
							DeniedResources: []ResourceSubresource{
								{
									Resource: "pods",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.resource = "pods"
					request.subresource = "create"
					request.namespace = "ns2"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedResourceManagedNamespace, "ns2", "pods"),
			},
		},
		"ClusterScopedVerbDenied_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name:        "",
							DeniedVerbs: []string{"verb1", "verb2", "verb3"},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb3"
					request.namespace = ""
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbClusterScopedResource, "verb3"),
			},
		},
		"ClusterScopedResourceDenied_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "",
							DeniedResources: []ResourceSubresource{
								{
									Resource:    "nodes",
									Subresource: "proxy",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.resource = "nodes"
					request.subresource = "proxy"
					request.namespace = ""
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedClusterScopedResource, "nodes/proxy"),
			},
		},
		"ClusterScopedSubresourceNotDenied_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "",
							DeniedResources: []ResourceSubresource{
								{
									Resource:    "nodes",
									Subresource: "proxy",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.resource = "nodes"
					request.subresource = "subres"
					request.namespace = ""
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"ResourceDeniedForNamespaceAndSubresourceEmpty_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "ns2",
							DeniedResources: []ResourceSubresource{
								{
									Resource: "pods",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.resource = "pods"
					request.subresource = ""
					request.namespace = "ns2"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedResourceManagedNamespace, "ns2", "pods"),
			},
		},
		"SubresourceDeniedForNamespace_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "ns2",
							DeniedResources: []ResourceSubresource{
								{
									Resource:    "pods",
									Subresource: "exec",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.resource = "pods"
					request.subresource = "exec"
					request.namespace = "ns2"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedSubresourceManagedNamespace, "ns2", "pods/exec"),
			},
		},
		"VerbAndSubresourceNotDeniedForNamespace_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedNamespaces = []ManagedNamespace{
						{
							Name: "ns2",
							DeniedResources: []ResourceSubresource{
								{
									Resource:    "pods",
									Subresource: "exec",
								},
							},
						},
					}
				},

				prepRequest: func() {
					request.verb = "verb4"
					request.resource = "pods"
					request.subresource = "log"
					request.namespace = "ns2"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"VerbNotAllowedForResource_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = "group.foo.bar"
					request.apiVersion = "v2"
					request.namespace = ""
					request.resource = "gadgets"
					request.name = "pillar-obj"

					request.verb = "delete"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbManagedResource, "gadgets/pillar-obj", "delete"),
			},
		},
		"VerbNotAllowedForSubresource_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = ""
					request.namespace = "kube-system"
					request.resource = "pods"
					request.subresource = "exec"
					request.name = "coolpod"

					request.verb = "someaction"
				},
			},
			expected{
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbManagedResource, "pods/coolpod/exec", "someaction"),
			},
		},
		"VerbAllowedForResource_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = "group.foo.bar"
					request.namespace = ""
					request.resource = "gadgets"
					request.name = "pillar-obj"

					request.verb = "get"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"VerbAllowedForSubresource_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = ""
					request.namespace = "kube-system"
					request.resource = "pods"
					request.subresource = "scale"
					request.name = "coolpod"

					request.verb = "edit"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"UnmanagedResource_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = "group.foo.bar"
					request.namespace = ""
					request.resource = "gadgets"
					request.name = "not-so-pillar"

					request.verb = "someaction"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"UnmanagedSubresource_NoOpinion": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = "group.foo.bar"
					request.namespace = ""
					request.resource = "gadgets"
					request.name = "not-so-pillar"
					request.subresource = "sub"

					request.verb = "someaction"
				},
			},
			expected{
				decision: authorizer.DecisionNoOpinion,
				reason:   authReasonNoOpinion,
			},
		},
		"ManagedResourceUnmanagedSubresource_IsDenied": {
			input{
				prepConfig: func(c *config) {
					c.ManagedResources = baseConfig.ManagedResources
				},
				prepRequest: func() {
					request.apiGroup = ""
					request.namespace = "kube-system"
					request.resource = "pods"
					request.subresource = "log" // 'log' subresource is not managed directly
					request.name = "coolpod"

					request.verb = "someaction"
				},
			},
			expected{
				// even though the subresource 'log' is not managed directly,
				// its parent resource is and the verb is not allowed
				decision: authorizer.DecisionDeny,
				reason:   fmt.Sprintf(authReasonDeniedVerbManagedResource, "pods/coolpod/log", "someaction"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			if runSelectedOnly && !tc.input.selected {
				return
			}

			config := &config{}
			request = &mockRequest{
				isResourceRequest: !tc.input.notResourceRequest,
				user:              mockUserInfo{},
			}

			if tc.input.prepConfig != nil {
				tc.input.prepConfig(config)
			}
			if tc.input.prepRequest != nil {
				tc.input.prepRequest()
			}

			peFlag := (atomicFlag)(1)
			if tc.input.policyEnforcementDisabled {
				peFlag = (atomicFlag)(0)
			}
			authz := &Authorizer{
				config:                    config,
				configHelper:              buildConfigHelper(config),
				policyEnforcerEnabledFlag: peFlag,
			}

			observedDecision, observedReason, observedErr := authz.Authorize(context.Background(), request)

			if diff := cmp.Diff(tc.expected.decision, observedDecision); diff != "" {
				t.Errorf("Authorize(...): -want decision, +got decision:\n%s", diff)
			}

			if diff := cmp.Diff(tc.expected.reason, observedReason); diff != "" {
				t.Errorf("Authorize(...): -want reason, +got reason:\n%s", diff)
			}

			var expectedErr error = nil
			if tc.expected.errFunc != nil {
				expectedErr = tc.expected.errFunc()
			}

			if diff := cmp.Diff(expectedErr, observedErr, cmp.Comparer(func(e1, e2 error) bool {
				if e1 == nil || e2 == nil {
					return e1 == nil && e2 == nil
				}
				return e1.Error() == e2.Error()
			})); diff != "" {
				t.Errorf("Authorize(...): -want error, +got error:\n%s", diff)
			}
		})
	}
}
