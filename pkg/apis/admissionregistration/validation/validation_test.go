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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func strPtr(s string) *string { return &s }

func int32Ptr(i int32) *int32 { return &i }

func newValidatingWebhookConfiguration(hooks []admissionregistration.ValidatingWebhook, defaultAdmissionReviewVersions bool) *admissionregistration.ValidatingWebhookConfiguration {
	// If the test case did not specify an AdmissionReviewVersions, default it so the test passes as
	// this field will be defaulted in production code.
	for i := range hooks {
		if defaultAdmissionReviewVersions && len(hooks[i].AdmissionReviewVersions) == 0 {
			hooks[i].AdmissionReviewVersions = []string{"v1beta1"}
		}
	}
	return &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Webhooks: hooks,
	}
}

func TestValidateValidatingWebhookConfiguration(t *testing.T) {
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		config        *admissionregistration.ValidatingWebhookConfiguration
		gv            schema.GroupVersion
		expectedError string
	}{
		{
			name: "AdmissionReviewVersions are required",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1beta1`,
		}, {
			name: "should fail on bad AdmissionReviewVersion value",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"0v"},
				},
			}, true),
			expectedError: `Invalid value: "0v": a DNS-1035 label`,
		},
		{
			name: "should pass on valid AdmissionReviewVersion",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1"},
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "should pass on mix of accepted and unaccepted AdmissionReviewVersion",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1", "invalid-version"},
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"invalidVersion"},
				},
			}, true),
			expectedError: `Invalid value: []string{"invalidVersion"}`,
		},
		{
			name: "should fail on duplicate AdmissionReviewVersion",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"v1beta1", "v1beta1"},
				},
			}, true),
			expectedError: `Invalid value: "v1beta1": duplicate version`,
		},
		{
			name: "all Webhooks must have a fully qualified name",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: `webhooks[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, webhooks[2].name: Required value`,
		},
		{
			name: "Webhooks must have unique names when not created via v1beta1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
		},
		{
			name: "Webhooks can have duplicate names when created via v1beta1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "Operations must not be empty or nil",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
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
			}, true),
			expectedError: `webhooks[0].rules[0].operations: Required value, webhooks[0].rules[1].operations: Required value`,
		},
		{
			name: "\"\" is NOT a valid operation",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
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
			}, true),
			expectedError: `Unsupported value: ""`,
		},
		{
			name: "operation must be either create/update/delete/connect",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
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
			}, true),
			expectedError: `Unsupported value: "PATCH"`,
		},
		{
			name: "wildcard operation cannot be mixed with other strings",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
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
			}, true),
			expectedError: `if '*' is present, must not specify other operations`,
		},
		{
			name: `resource "*" can co-exist with resources that have subresources`,
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
		{
			name: `resource "*" cannot mix with resources that don't have subresources`,
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `if '*' is present, must not specify other resources without subresources`,
		},
		{
			name: "resource a/* cannot mix with a/x",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
		},
		{
			name: "resource a/* can mix with a",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
		{
			name: "resource */a cannot mix with x/a",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
		},
		{
			name: "resource */* cannot mix with other resources",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
		},
		{
			name: "FailurePolicy can only be \"Ignore\" or \"Fail\"",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
					FailurePolicy: func() *admissionregistration.FailurePolicyType {
						r := admissionregistration.FailurePolicyType("other")
						return &r
					}(),
				},
			}, true),
			expectedError: `webhooks[0].failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
		},
		{
			name: "AdmissionReviewVersions are required",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1beta1`,
		},
		{
			name: "SideEffects are required",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  nil,
				},
			}, true),
			expectedError: `webhooks[0].sideEffects: Required value: must specify one of None, NoneOnDryRun`,
		},
		{
			name: "SideEffects can only be \"Unknown\", \"None\", \"Some\", or \"NoneOnDryRun\" via v1beta1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects: func() *admissionregistration.SideEffectClass {
						r := admissionregistration.SideEffectClass("other")
						return &r
					}(),
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun", "Some", "Unknown"`,
		},
		{
			name: "SideEffects can only be \"None\" or \"NoneOnDryRun\" via v1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects: func() *admissionregistration.SideEffectClass {
						r := admissionregistration.SideEffectClass("other")
						return &r
					}(),
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1"},
			expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun"`,
		},
		{
			name: "both service and URL missing",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{},
				},
			}, true),
			expectedError: `exactly one of`,
		},
		{
			name: "both service and URL provided",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Port:      443,
						},
						URL: strPtr("example.com/k8s/webhook"),
					},
				},
			}, true),
			expectedError: `[0].clientConfig: Required value: exactly one of url or service is required`,
		},
		{
			name: "blank URL",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr(""),
					},
				},
			}, true),
			expectedError: `[0].clientConfig.url: Invalid value: "": host must be provided`,
		},
		{
			name: "wrong scheme",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("http://example.com"),
					},
				},
			}, true),
			expectedError: `https`,
		},
		{
			name: "missing host",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https:///fancy/webhook"),
					},
				},
			}, true),
			expectedError: `host must be provided`,
		},
		{
			name: "fragment",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://example.com/#bookmark"),
					},
				},
			}, true),
			expectedError: `"bookmark": fragments are not permitted`,
		},
		{
			name: "query",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://example.com?arg=value"),
					},
				},
			}, true),
			expectedError: `"arg=value": query parameters are not permitted`,
		},
		{
			name: "user",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://harry.potter@example.com/"),
					},
				},
			}, true),
			expectedError: `"harry.potter": user information is not permitted`,
		},
		{
			name: "just totally wrong",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("arg#backwards=thisis?html.index/port:host//:https"),
					},
				},
			}, true),
			expectedError: `host must be provided`,
		},
		{
			name: "path must start with slash",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("foo/"),
							Port:      443,
						},
					},
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "foo/": must start with a '/'`,
		},
		{
			name: "path accepts slash",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "path accepts no trailing slash",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "path fails //",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("//"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "//": segment[0] may not be empty`,
		},
		{
			name: "path no empty step",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo//bar/"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/foo//bar/": segment[1] may not be empty`,
		}, {
			name: "path no empty step 2",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo/bar//"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/foo/bar//": segment[2] may not be empty`,
		},
		{
			name: "path no non-subdomain",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/apis/foo.bar/v1alpha1/--bad"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/apis/foo.bar/v1alpha1/--bad": segment[3]: a DNS-1123 subdomain`,
		},
		{
			name: "invalid port 0",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.ValidatingWebhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							Service: &admissionregistration.ServiceReference{
								Namespace: "ns",
								Name:      "n",
								Path:      strPtr("https://apis/foo.bar"),
								Port:      0,
							},
						},
						SideEffects: &unknownSideEffect,
					},
				}, true),
			expectedError: `Invalid value: 0: port is not valid: must be between 1 and 65535, inclusive`,
		},
		{
			name: "invalid port >65535",
			config: newValidatingWebhookConfiguration(
				[]admissionregistration.ValidatingWebhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							Service: &admissionregistration.ServiceReference{
								Namespace: "ns",
								Name:      "n",
								Path:      strPtr("https://apis/foo.bar"),
								Port:      65536,
							},
						},
						SideEffects: &unknownSideEffect,
					},
				}, true),
			expectedError: `Invalid value: 65536: port is not valid: must be between 1 and 65535, inclusive`,
		},
		{
			name: "timeout seconds cannot be greater than 30",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(31),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: 31: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "timeout seconds cannot be smaller than 1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(0),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: 0: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "timeout seconds must be positive",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(-1),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: -1: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "valid timeout seconds",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(1),
				},
				{
					Name:           "webhook2.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(15),
				},
				{
					Name:           "webhook3.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(30),
				},
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingWebhookConfiguration(test.config, test.gv)
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

func TestValidateValidatingWebhookConfigurationUpdate(t *testing.T) {
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.ValidatingWebhookConfiguration
		config        *admissionregistration.ValidatingWebhookConfiguration
		gv            schema.GroupVersion
		expectedError string
	}{
		{
			name: "should pass on valid new AdmissionReviewVersion",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1"},
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			expectedError: ``,
		},
		{
			name: "should pass on invalid AdmissionReviewVersion with invalid previous versions",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1", "invalid-v2"},
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v0"},
				},
			}, true),
			expectedError: ``,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion with valid previous versions",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1"},
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1", "invalid-v1"},
				},
			}, true),
			expectedError: `Invalid value: []string{"invalid-v1"}`,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion with missing previous versions",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1"},
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `Invalid value: []string{"invalid-v1"}`,
		},
		{
			name: "Webhooks must have unique names when not updated via v1beta1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
		},
		{
			name: "Webhooks can have duplicate names when old config has duplicate names",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: ``,
		},
		{
			name: "Webhooks can have duplicate names when updated via v1beta1",
			config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingWebhookConfigurationUpdate(test.config, test.oldconfig, test.gv)
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

func newMutatingWebhookConfiguration(hooks []admissionregistration.MutatingWebhook, defaultAdmissionReviewVersions bool) *admissionregistration.MutatingWebhookConfiguration {
	// If the test case did not specify an AdmissionReviewVersions, default it so the test passes as
	// this field will be defaulted in production code.
	for i := range hooks {
		if defaultAdmissionReviewVersions && len(hooks[i].AdmissionReviewVersions) == 0 {
			hooks[i].AdmissionReviewVersions = []string{"v1beta1"}
		}
	}
	return &admissionregistration.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Webhooks: hooks,
	}
}

func TestValidateMutatingWebhookConfiguration(t *testing.T) {
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		config        *admissionregistration.MutatingWebhookConfiguration
		gv            schema.GroupVersion
		expectedError string
	}{
		{
			name: "AdmissionReviewVersions are required",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1beta1`,
		}, {
			name: "should fail on bad AdmissionReviewVersion value",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"0v"},
				},
			}, true),
			expectedError: `Invalid value: "0v": a DNS-1035 label`,
		},
		{
			name: "should pass on valid AdmissionReviewVersion",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1"},
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "should pass on mix of accepted and unaccepted AdmissionReviewVersion",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1", "invalid-version"},
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"invalidVersion"},
				},
			}, true),
			expectedError: `Invalid value: []string{"invalidVersion"}`,
		},
		{
			name: "should fail on duplicate AdmissionReviewVersion",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					AdmissionReviewVersions: []string{"v1beta1", "v1beta1"},
				},
			}, true),
			expectedError: `Invalid value: "v1beta1": duplicate version`,
		},
		{
			name: "all Webhooks must have a fully qualified name",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: `webhooks[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, webhooks[2].name: Required value`,
		},
		{
			name: "Webhooks must have unique names when not created via v1beta1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
		},
		{
			name: "Webhooks can have duplicate names when created via v1beta1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "Operations must not be empty or nil",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
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
			}, true),
			expectedError: `webhooks[0].rules[0].operations: Required value, webhooks[0].rules[1].operations: Required value`,
		},
		{
			name: "\"\" is NOT a valid operation",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
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
			}, true),
			expectedError: `Unsupported value: ""`,
		},
		{
			name: "operation must be either create/update/delete/connect",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
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
			}, true),
			expectedError: `Unsupported value: "PATCH"`,
		},
		{
			name: "wildcard operation cannot be mixed with other strings",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
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
			}, true),
			expectedError: `if '*' is present, must not specify other operations`,
		},
		{
			name: `resource "*" can co-exist with resources that have subresources`,
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
		{
			name: `resource "*" cannot mix with resources that don't have subresources`,
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `if '*' is present, must not specify other resources without subresources`,
		},
		{
			name: "resource a/* cannot mix with a/x",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
		},
		{
			name: "resource a/* can mix with a",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
		{
			name: "resource */a cannot mix with x/a",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
		},
		{
			name: "resource */* cannot mix with other resources",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
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
			}, true),
			expectedError: `webhooks[0].rules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
		},
		{
			name: "FailurePolicy can only be \"Ignore\" or \"Fail\"",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
					FailurePolicy: func() *admissionregistration.FailurePolicyType {
						r := admissionregistration.FailurePolicyType("other")
						return &r
					}(),
				},
			}, true),
			expectedError: `webhooks[0].failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
		},
		{
			name: "AdmissionReviewVersions are required",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1beta1`,
		},
		{
			name: "SideEffects are required",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  nil,
				},
			}, true),
			expectedError: `webhooks[0].sideEffects: Required value: must specify one of None, NoneOnDryRun`,
		},
		{
			name: "SideEffects can only be \"Unknown\", \"None\", \"Some\", or \"NoneOnDryRun\" via v1beta1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects: func() *admissionregistration.SideEffectClass {
						r := admissionregistration.SideEffectClass("other")
						return &r
					}(),
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun", "Some", "Unknown"`,
		},
		{
			name: "SideEffects can only be \"None\" or \"NoneOnDryRun\" via v1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects: func() *admissionregistration.SideEffectClass {
						r := admissionregistration.SideEffectClass("other")
						return &r
					}(),
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1"},
			expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun"`,
		},
		{
			name: "both service and URL missing",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{},
				},
			}, true),
			expectedError: `exactly one of`,
		},
		{
			name: "both service and URL provided",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Port:      443,
						},
						URL: strPtr("example.com/k8s/webhook"),
					},
				},
			}, true),
			expectedError: `[0].clientConfig: Required value: exactly one of url or service is required`,
		},
		{
			name: "blank URL",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr(""),
					},
				},
			}, true),
			expectedError: `[0].clientConfig.url: Invalid value: "": host must be provided`,
		},
		{
			name: "wrong scheme",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("http://example.com"),
					},
				},
			}, true),
			expectedError: `https`,
		},
		{
			name: "missing host",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https:///fancy/webhook"),
					},
				},
			}, true),
			expectedError: `host must be provided`,
		},
		{
			name: "fragment",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://example.com/#bookmark"),
					},
				},
			}, true),
			expectedError: `"bookmark": fragments are not permitted`,
		},
		{
			name: "query",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://example.com?arg=value"),
					},
				},
			}, true),
			expectedError: `"arg=value": query parameters are not permitted`,
		},
		{
			name: "user",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("https://harry.potter@example.com/"),
					},
				},
			}, true),
			expectedError: `"harry.potter": user information is not permitted`,
		},
		{
			name: "just totally wrong",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						URL: strPtr("arg#backwards=thisis?html.index/port:host//:https"),
					},
				},
			}, true),
			expectedError: `host must be provided`,
		},
		{
			name: "path must start with slash",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("foo/"),
							Port:      443,
						},
					},
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "foo/": must start with a '/'`,
		},
		{
			name: "path accepts slash",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "path accepts no trailing slash",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "path fails //",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("//"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "//": segment[0] may not be empty`,
		},
		{
			name: "path no empty step",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo//bar/"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/foo//bar/": segment[1] may not be empty`,
		}, {
			name: "path no empty step 2",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/foo/bar//"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/foo/bar//": segment[2] may not be empty`,
		},
		{
			name: "path no non-subdomain",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name: "webhook.k8s.io",
					ClientConfig: admissionregistration.WebhookClientConfig{
						Service: &admissionregistration.ServiceReference{
							Namespace: "ns",
							Name:      "n",
							Path:      strPtr("/apis/foo.bar/v1alpha1/--bad"),
							Port:      443,
						},
					},
					SideEffects: &unknownSideEffect,
				},
			}, true),
			expectedError: `clientConfig.service.path: Invalid value: "/apis/foo.bar/v1alpha1/--bad": segment[3]: a DNS-1123 subdomain`,
		},
		{
			name: "invalid port 0",
			config: newMutatingWebhookConfiguration(
				[]admissionregistration.MutatingWebhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							Service: &admissionregistration.ServiceReference{
								Namespace: "ns",
								Name:      "n",
								Path:      strPtr("https://apis/foo.bar"),
								Port:      0,
							},
						},
						SideEffects: &unknownSideEffect,
					},
				}, true),
			expectedError: `Invalid value: 0: port is not valid: must be between 1 and 65535, inclusive`,
		},
		{
			name: "invalid port >65535",
			config: newMutatingWebhookConfiguration(
				[]admissionregistration.MutatingWebhook{
					{
						Name: "webhook.k8s.io",
						ClientConfig: admissionregistration.WebhookClientConfig{
							Service: &admissionregistration.ServiceReference{
								Namespace: "ns",
								Name:      "n",
								Path:      strPtr("https://apis/foo.bar"),
								Port:      65536,
							},
						},
						SideEffects: &unknownSideEffect,
					},
				}, true),
			expectedError: `Invalid value: 65536: port is not valid: must be between 1 and 65535, inclusive`,
		},
		{
			name: "timeout seconds cannot be greater than 30",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(31),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: 31: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "timeout seconds cannot be smaller than 1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(0),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: 0: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "timeout seconds must be positive",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(-1),
				},
			}, true),
			expectedError: `webhooks[0].timeoutSeconds: Invalid value: -1: the timeout value must be between 1 and 30 seconds`,
		},
		{
			name: "valid timeout seconds",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:           "webhook.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(1),
				},
				{
					Name:           "webhook2.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(15),
				},
				{
					Name:           "webhook3.k8s.io",
					ClientConfig:   validClientConfig,
					SideEffects:    &unknownSideEffect,
					TimeoutSeconds: int32Ptr(30),
				},
			}, true),
			gv: schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingWebhookConfiguration(test.config, test.gv)
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

func TestValidateMutatingWebhookConfigurationUpdate(t *testing.T) {
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	noSideEffect := admissionregistration.SideEffectClassNone
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.MutatingWebhookConfiguration
		config        *admissionregistration.MutatingWebhookConfiguration
		gv            schema.GroupVersion
		expectedError string
	}{
		{
			name: "should pass on valid new AdmissionReviewVersion",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1"},
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			expectedError: ``,
		},
		{
			name: "should pass on invalid AdmissionReviewVersion with invalid previous versions",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1", "invalid-v2"},
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v0"},
				},
			}, true),
			expectedError: ``,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion with valid previous versions",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1"},
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"v1beta1", "invalid-v1"},
				},
			}, true),
			expectedError: `Invalid value: []string{"invalid-v1"}`,
		},
		{
			name: "should fail on invalid AdmissionReviewVersion with missing previous versions",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:                    "webhook.k8s.io",
					ClientConfig:            validClientConfig,
					SideEffects:             &unknownSideEffect,
					AdmissionReviewVersions: []string{"invalid-v1"},
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			expectedError: `Invalid value: []string{"invalid-v1"}`,
		},
		{
			name: "Webhooks can have duplicate names when old config has duplicate names",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: ``,
		},
		{
			name: "Webhooks can have duplicate names when updated via v1beta1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, false),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
		{
			name: "Webhooks can't have side effects when old config has no side effects via v1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &noSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1"},
			expectedError: `Unsupported value: "Unknown": supported values: "None", "NoneOnDryRun"`,
		},
		{
			name: "Webhooks can have side effects when old config has side effects",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			gv:            schema.GroupVersion{Group: "foo", Version: "bar"},
			expectedError: ``,
		},
		{
			name: "Webhooks can have side effects when updated via v1beta1",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &unknownSideEffect,
				},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{
				{
					Name:         "webhook.k8s.io",
					ClientConfig: validClientConfig,
					SideEffects:  &noSideEffect,
				},
			}, false),
			gv:            schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
			expectedError: ``,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingWebhookConfigurationUpdate(test.config, test.oldconfig, test.gv)
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
