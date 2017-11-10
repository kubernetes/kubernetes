/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func getInitializerConfiguration(initializers []admissionregistration.Initializer) *admissionregistration.InitializerConfiguration {
	return &admissionregistration.InitializerConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Initializers: initializers,
	}
}

func TestValidateInitializerConfiguration(t *testing.T) {
	tests := []struct {
		name          string
		config        *admissionregistration.InitializerConfiguration
		expectedError string
	}{
		{
			name: "0 rule is valid",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
					},
				}),
		},
		{
			name: "all initializers must have a fully qualified name",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
					},
					{
						Name: "k8s.io",
					},
					{
						Name: "",
					},
				}),
			expectedError: `initializers[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, initializers[2].name: Required value`,
		},
		{
			name: "APIGroups must not be empty or nil",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
							{
								APIGroups:   nil,
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				}),
			expectedError: `initializers[0].rules[0].apiGroups: Required value, initializers[0].rules[1].apiGroups: Required value`,
		},
		{
			name: "APIVersions must not be empty or nil",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{},
								Resources:   []string{"a"},
							},
							{
								APIGroups:   []string{"a"},
								APIVersions: nil,
								Resources:   []string{"a"},
							},
						},
					},
				}),
			expectedError: `initializers[0].rules[0].apiVersions: Required value, initializers[0].rules[1].apiVersions: Required value`,
		},
		{
			name: "Resources must not be empty or nil",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{},
							},
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   nil,
							},
						},
					},
				}),
			expectedError: `initializers[0].rules[0].resources: Required value, initializers[0].rules[1].resources: Required value`,
		},
		{
			name: "\"\" is a valid APIGroup",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a", ""},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				}),
		},
		{
			name: "\"\" is NOT a valid APIVersion",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a", ""},
								Resources:   []string{"a"},
							},
						},
					},
				}),
			expectedError: "apiVersions[1]: Required value",
		},
		{
			name: "\"\" is NOT a valid Resource",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a", ""},
							},
						},
					},
				}),
			expectedError: "resources[1]: Required value",
		},
		{
			name: "wildcard cannot be mixed with other strings for APIGroups or APIVersions or Resources",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a", "*"},
								APIVersions: []string{"a", "*"},
								Resources:   []string{"a", "*"},
							},
						},
					},
				}),
			expectedError: `[initializers[0].rules[0].apiGroups: Invalid value: []string{"a", "*"}: if '*' is present, must not specify other API groups, initializers[0].rules[0].apiVersions: Invalid value: []string{"a", "*"}: if '*' is present, must not specify other API versions, initializers[0].rules[0].resources: Invalid value: []string{"a", "*"}: if '*' is present, must not specify other resources]`,
		},
		{
			name: "Subresource not allowed",
			config: getInitializerConfiguration(
				[]admissionregistration.Initializer{
					{
						Name: "initializer.k8s.io",
						Rules: []admissionregistration.Rule{
							{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/b"},
							},
						},
					},
				}),
			expectedError: ` "a/b": must not specify subresources`,
		},
	}

	for _, test := range tests {
		errs := ValidateInitializerConfiguration(test.config)
		err := errs.ToAggregate()
		if err != nil {
			if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
				t.Errorf("test case %s, expected to contain %s, got %s", test.name, e, a)
			}
		} else {
			if test.expectedError != "" {
				t.Errorf("test case %s, unexpected no error, expected to contain %s", test.name, test.expectedError)
			}
		}
	}
}

func newValidatingWebhookConfiguration(hooks []admissionregistration.Webhook) *admissionregistration.ValidatingWebhookConfiguration {
	return &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Webhooks: hooks,
	}
}

// TODO: Add TestValidateMutatingWebhookConfiguration to test validation for mutating webhooks.

func TestValidateValidatingWebhookConfiguration(t *testing.T) {
	tests := []struct {
		name          string
		config        *admissionregistration.ValidatingWebhookConfiguration
		expectedError string
	}{
		{
			name: "all Webhooks must have a fully qualified name",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
					},
					{
						Name: "k8s.io",
					},
					{
						Name: "",
					},
				}),
			expectedError: `webhooks[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, webhooks[2].name: Required value`,
		},
		{
			name: "Operations must not be empty or nil",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a"},
								},
							},
							{
								Operations: nil,
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a"},
								},
							},
						},
					},
				}),
			expectedError: `webhooks[0].rules[0].operations: Required value, webhooks[0].rules[1].operations: Required value`,
		},
		{
			name: "\"\" is NOT a valid operation",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE", ""},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a"},
								},
							},
						},
					},
				}),
			expectedError: `Unsupported value: ""`,
		},
		{
			name: "operation must be either create/update/delete/connect",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"PATCH"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a"},
								},
							},
						},
					},
				}),
			expectedError: `Unsupported value: "PATCH"`,
		},
		{
			name: "wildcard operation cannot be mixed with other strings",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE", "*"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a"},
								},
							},
						},
					},
				}),
			expectedError: `if '*' is present, must not specify other operations`,
		},
		{
			name: `resource "*" can co-exist with resources that have subresources`,
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"*", "a/b", "a/*", "*/b"},
								},
							},
						},
					},
				}),
		},
		{
			name: `resource "*" cannot mix with resources that don't have subresources`,
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"*", "a"},
								},
							},
						},
					},
				}),
			expectedError: `if '*' is present, must not specify other resources without subresources`,
		},
		{
			name: "resource a/* cannot mix with a/x",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a/*", "a/x"},
								},
							},
						},
					},
				}),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
		},
		{
			name: "resource a/* can mix with a",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"a/*", "a"},
								},
							},
						},
					},
				}),
		},
		{
			name: "resource */a cannot mix with x/a",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"*/a", "x/a"},
								},
							},
						},
					},
				}),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
		},
		{
			name: "resource */* cannot mix with other resources",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						Rules: []admissionregistration.RuleWithOperations{
							{
								Operations: []admissionregistration.OperationType{"CREATE"},
								Rule: admissionregistration.Rule{
									APIGroups:   []string{"a"},
									APIVersions: []string{"a"},
									Resources:   []string{"*/*", "a"},
								},
							},
						},
					},
				}),
			expectedError: `webhooks[0].rules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
		},
		{
			name: "FailurePolicy can only be \"Ignore\" or \"Fail\"",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						FailurePolicy: func() *admissionregistration.FailurePolicyType {
							r := admissionregistration.FailurePolicyType("other")
							return &r
						}(),
					},
				}),
			expectedError: `webhooks[0].failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
		},
		{
			name: "URLPath must start with slash",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "foo/",
						},
					},
				}),
			expectedError: `clientConfig.urlPath: Invalid value: "foo/": must start with a '/'`,
		},
		{
			name: "URLPath accepts slash",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "/",
						},
					},
				}),
			expectedError: ``,
		},
		{
			name: "URLPath accepts no trailing slash",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "/foo",
						},
					},
				}),
			expectedError: ``,
		},
		{
			name: "URLPath fails //",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "//",
						},
					},
				}),
			expectedError: `clientConfig.urlPath: Invalid value: "//": segment[0] may not be empty`,
		},
		{
			name: "URLPath no empty step",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "/foo//bar/",
						},
					},
				}),
			expectedError: `clientConfig.urlPath: Invalid value: "/foo//bar/": segment[1] may not be empty`,
		}, {
			name: "URLPath no empty step 2",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "/foo/bar//",
						},
					},
				}),
			expectedError: `clientConfig.urlPath: Invalid value: "/foo/bar//": segment[2] may not be empty`,
		},
		{
			name: "URLPath no non-subdomain",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.Webhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							URLPath: "/apis/foo.bar/v1alpha1/--bad",
						},
					},
				}),
			expectedError: `clientConfig.urlPath: Invalid value: "/apis/foo.bar/v1alpha1/--bad": segment[3]: a DNS-1123 subdomain`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingWebhookConfiguration(test.config)
			err := errs.ToAggregate()
			if err != nil {
				if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain %s, got %s", e, a)
				}
			} else {
				if test.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain %s", test.expectedError)
				}
			}
		})

	}
}
