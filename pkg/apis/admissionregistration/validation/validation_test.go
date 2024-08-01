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
	"fmt"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func ptr[T any](v T) *T { return &v }

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
	noSideEffect := admissionregistration.SideEffectClassNone
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		config        *admissionregistration.ValidatingWebhookConfiguration
		expectedError string
	}{{
		name: "AdmissionReviewVersions are required",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1, v1beta1`,
	}, {
		name: "should fail on bad AdmissionReviewVersion value",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"0v"},
		},
		}, true),
		expectedError: `Invalid value: "0v": a DNS-1035 label`,
	}, {
		name: "should pass on valid AdmissionReviewVersion",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &noSideEffect,
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should pass on mix of accepted and unaccepted AdmissionReviewVersion",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &noSideEffect,
			AdmissionReviewVersions: []string{"v1beta1", "invalid-version"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should fail on invalid AdmissionReviewVersion",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"invalidVersion"},
		},
		}, true),
		expectedError: `Invalid value: []string{"invalidVersion"}`,
	}, {
		name: "should fail on duplicate AdmissionReviewVersion",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"v1beta1", "v1beta1"},
		},
		}, true),
		expectedError: `Invalid value: "v1beta1": duplicate version`,
	}, {
		name: "all Webhooks must have a fully qualified name",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		}, {
			Name:         "k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		}, {
			Name:         "",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		},
		}, true),
		expectedError: `webhooks[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, webhooks[2].name: Required value`,
	}, {
		name: "Webhooks must have unique names when created",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
	}, {
		name: "Operations must not be empty or nil",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}, {
				Operations: nil,
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].operations: Required value, webhooks[0].rules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE", ""},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"PATCH"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE", "*"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*", "a/b", "a/*", "*/b"},
				},
			}},
		},
		}, true),
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*", "a"},
				},
			}},
		},
		}, true),
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a/*", "a/x"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a/*", "a"},
				},
			}},
		},
		}, true),
	}, {
		name: "resource */a cannot mix with x/a",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*/a", "x/a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*/*", "a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "FailurePolicy can only be \"Ignore\" or \"Fail\"",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "AdmissionReviewVersions are required",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1, v1beta1`,
	}, {
		name: "SideEffects are required",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  nil,
		},
		}, true),
		expectedError: `webhooks[0].sideEffects: Required value: must specify one of None, NoneOnDryRun`,
	}, {
		name: "SideEffects can only be \"None\" or \"NoneOnDryRun\" when created",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects: func() *admissionregistration.SideEffectClass {
				r := admissionregistration.SideEffectClass("other")
				return &r
			}(),
		},
		}, true),
		expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun"`,
	}, {
		name: "both service and URL missing",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{},
		},
		}, true),
		expectedError: `exactly one of`,
	}, {
		name: "both service and URL provided",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "blank URL",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr(""),
			},
		},
		}, true),
		expectedError: `[0].clientConfig.url: Invalid value: "": host must be specified`,
	}, {
		name: "wrong scheme",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("http://example.com"),
			},
		},
		}, true),
		expectedError: `https`,
	}, {
		name: "missing host",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https:///fancy/webhook"),
			},
		},
		}, true),
		expectedError: `host must be specified`,
	}, {
		name: "fragment",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://example.com/#bookmark"),
			},
		},
		}, true),
		expectedError: `"bookmark": fragments are not permitted`,
	}, {
		name: "query",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://example.com?arg=value"),
			},
		},
		}, true),
		expectedError: `"arg=value": query parameters are not permitted`,
	}, {
		name: "user",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://harry.potter@example.com/"),
			},
		},
		}, true),
		expectedError: `"harry.potter": user information is not permitted`,
	}, {
		name: "just totally wrong",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("arg#backwards=thisis?html.index/port:host//:https"),
			},
		},
		}, true),
		expectedError: `host must be specified`,
	}, {
		name: "path must start with slash",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "path accepts slash",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("/"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "path accepts no trailing slash",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("/foo"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "path fails //",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("//"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: `clientConfig.service.path: Invalid value: "//": segment[0] may not be empty`,
	}, {
		name: "path no empty step",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "path no non-subdomain",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
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
		expectedError: `clientConfig.service.path: Invalid value: "/apis/foo.bar/v1alpha1/--bad": segment[3]: a lowercase RFC 1123 subdomain`,
	}, {
		name: "invalid port 0",
		config: newValidatingWebhookConfiguration(
			[]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "invalid port >65535",
		config: newValidatingWebhookConfiguration(
			[]admissionregistration.ValidatingWebhook{{
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
	}, {
		name: "timeout seconds cannot be greater than 30",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(31),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: 31: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "timeout seconds cannot be smaller than 1",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(0),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: 0: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "timeout seconds must be positive",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(-1),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: -1: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "valid timeout seconds",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(1),
		}, {
			Name:           "webhook2.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(15),
		}, {
			Name:           "webhook3.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(30),
		},
		}, true),
	}, {
		name: "single match condition must have a name",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Required value`,
	}, {
		name: "all match conditions must have a name",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Expression: "true",
			}, {
				Expression: "true",
			}},
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Required value, webhooks[0].matchConditions[1].name: Required value, webhooks[1].matchConditions[0].name: Required value`,
	}, {
		name: "single match condition must have a qualified name",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "-hello",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Invalid value: "-hello": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
	}, {
		name: "all match conditions must have qualified names",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       ".io",
				Expression: "true",
			}, {
				Name:       "thing.test.com",
				Expression: "true",
			}},
		}, {
			Name:         "webhook2.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "some name",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `[webhooks[0].matchConditions[0].name: Invalid value: ".io": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]'), webhooks[1].matchConditions[0].name: Invalid value: "some name": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')]`,
	}, {
		name: "expression is required",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name: "webhook.k8s.io",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Required value`,
	}, {
		name: "expression is required to have some value",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Required value`,
	}, {
		name: "invalid expression",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "object.x in [1, 2, ",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Invalid value: "object.x in [1, 2,": compilation failed: ERROR: <input>:1:19: Syntax error: missing ']' at '<EOF>'`,
	}, {
		name: "unique names same hook",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}, {
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `matchConditions[1].name: Duplicate value: "webhook.k8s.io"`,
	}, {
		name: "repeat names allowed across different hooks",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		}, {
			Name:         "webhook2.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "must evaluate to bool",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "6",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Invalid value: "6": must evaluate to bool`,
	}, {
		name: "max of 64 match conditions",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:            "webhook.k8s.io",
			ClientConfig:    validClientConfig,
			SideEffects:     &noSideEffect,
			MatchConditions: get65MatchConditions(),
		},
		}, true),
		expectedError: `webhooks[0].matchConditions: Too many: 65: must have at most 64 items`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingWebhookConfiguration(test.config)
			err := errs.ToAggregate()
			if err != nil {
				if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain:\n  %s\ngot:\n  %s", e, a)
				}
			} else {
				if test.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain:\n  %s", test.expectedError)
				}
			}
		})

	}
}

func TestValidateValidatingWebhookConfigurationUpdate(t *testing.T) {
	noSideEffect := admissionregistration.SideEffectClassNone
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.ValidatingWebhookConfiguration
		config        *admissionregistration.ValidatingWebhookConfiguration
		expectedError string
	}{{
		name: "should pass on valid new AdmissionReviewVersion",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should pass on invalid AdmissionReviewVersion with invalid previous versions",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1", "invalid-v2"},
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v0"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should fail on invalid AdmissionReviewVersion with valid previous versions",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1"},
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"v1beta1", "invalid-v1"},
		},
		}, true),
		expectedError: `Invalid value: []string{"invalid-v1"}`,
	}, {
		name: "should fail on invalid AdmissionReviewVersion with missing previous versions",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1"},
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `Invalid value: []string{"invalid-v1"}`,
	}, {
		name: "Webhooks must have unique names when old config has unique names",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
	}, {
		name: "Webhooks can have duplicate names when old config has duplicate names",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "Webhooks must compile CEL expressions with StoredExpression environment if unchanged",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "test() == true",
			}},
		},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "test() == true",
			}}},
		}, true),
	}, {
		name: "Webhooks must compile CEL expressions with NewExpression environment type if changed",
		config: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "test() == true",
			}}},
		}, true),
		oldconfig: newValidatingWebhookConfiguration([]admissionregistration.ValidatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "true",
			}}},
		}, true),
		expectedError: `undeclared reference to 'test'`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingWebhookConfigurationUpdate(test.config, test.oldconfig)
			err := errs.ToAggregate()
			if err != nil {
				if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain:\n  %s\ngot:\n  %s", e, a)
				}
			} else {
				if test.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain:\n  %s", test.expectedError)
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
	noSideEffect := admissionregistration.SideEffectClassNone
	unknownSideEffect := admissionregistration.SideEffectClassUnknown
	validClientConfig := admissionregistration.WebhookClientConfig{
		URL: strPtr("https://example.com"),
	}
	tests := []struct {
		name          string
		config        *admissionregistration.MutatingWebhookConfiguration
		expectedError string
	}{{
		name: "AdmissionReviewVersions are required",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1, v1beta1`,
	}, {
		name: "should fail on bad AdmissionReviewVersion value",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"0v"},
		},
		}, true),
		expectedError: `Invalid value: "0v": a DNS-1035 label`,
	}, {
		name: "should pass on valid AdmissionReviewVersion",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &noSideEffect,
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should pass on mix of accepted and unaccepted AdmissionReviewVersion",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &noSideEffect,
			AdmissionReviewVersions: []string{"v1beta1", "invalid-version"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should fail on invalid AdmissionReviewVersion",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"invalidVersion"},
		},
		}, true),
		expectedError: `Invalid value: []string{"invalidVersion"}`,
	}, {
		name: "should fail on duplicate AdmissionReviewVersion",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			AdmissionReviewVersions: []string{"v1beta1", "v1beta1"},
		},
		}, true),
		expectedError: `Invalid value: "v1beta1": duplicate version`,
	}, {
		name: "all Webhooks must have a fully qualified name",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		}, {
			Name:         "k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		}, {
			Name:         "",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		},
		}, true),
		expectedError: `webhooks[1].name: Invalid value: "k8s.io": should be a domain with at least three segments separated by dots, webhooks[2].name: Required value`,
	}, {
		name: "Webhooks must have unique names when created",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: `webhooks[1].name: Duplicate value: "webhook.k8s.io"`,
	}, {
		name: "Operations must not be empty or nil",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}, {
				Operations: nil,
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].operations: Required value, webhooks[0].rules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE", ""},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"PATCH"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE", "*"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a"},
				},
			}},
		},
		}, true),
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*", "a/b", "a/*", "*/b"},
				},
			}},
		},
		}, true),
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*", "a"},
				},
			}},
		},
		}, true),
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a/*", "a/x"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a/*", "a"},
				},
			}},
		},
		}, true),
	}, {
		name: "resource */a cannot mix with x/a",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*/a", "x/a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
			Rules: []admissionregistration.RuleWithOperations{{
				Operations: []admissionregistration.OperationType{"CREATE"},
				Rule: admissionregistration.Rule{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"*/*", "a"},
				},
			}},
		},
		}, true),
		expectedError: `webhooks[0].rules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "FailurePolicy can only be \"Ignore\" or \"Fail\"",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "AdmissionReviewVersions are required",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `webhooks[0].admissionReviewVersions: Required value: must specify one of v1, v1beta1`,
	}, {
		name: "SideEffects are required",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  nil,
		},
		}, true),
		expectedError: `webhooks[0].sideEffects: Required value: must specify one of None, NoneOnDryRun`,
	}, {
		name: "SideEffects can only be \"None\" or \"NoneOnDryRun\" when created",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects: func() *admissionregistration.SideEffectClass {
				r := admissionregistration.SideEffectClass("other")
				return &r
			}(),
		},
		}, true),
		expectedError: `webhooks[0].sideEffects: Unsupported value: "other": supported values: "None", "NoneOnDryRun"`,
	}, {
		name: "both service and URL missing",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{},
		},
		}, true),
		expectedError: `exactly one of`,
	}, {
		name: "both service and URL provided",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "blank URL",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr(""),
			},
		},
		}, true),
		expectedError: `[0].clientConfig.url: Invalid value: "": host must be specified`,
	}, {
		name: "wrong scheme",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("http://example.com"),
			},
		},
		}, true),
		expectedError: `https`,
	}, {
		name: "missing host",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https:///fancy/webhook"),
			},
		},
		}, true),
		expectedError: `host must be specified`,
	}, {
		name: "fragment",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://example.com/#bookmark"),
			},
		},
		}, true),
		expectedError: `"bookmark": fragments are not permitted`,
	}, {
		name: "query",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://example.com?arg=value"),
			},
		},
		}, true),
		expectedError: `"arg=value": query parameters are not permitted`,
	}, {
		name: "user",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("https://harry.potter@example.com/"),
			},
		},
		}, true),
		expectedError: `"harry.potter": user information is not permitted`,
	}, {
		name: "just totally wrong",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				URL: strPtr("arg#backwards=thisis?html.index/port:host//:https"),
			},
		},
		}, true),
		expectedError: `host must be specified`,
	}, {
		name: "path must start with slash",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "path accepts slash",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("/"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "path accepts no trailing slash",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("/foo"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "path fails //",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name: "webhook.k8s.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Namespace: "ns",
					Name:      "n",
					Path:      strPtr("//"),
					Port:      443,
				},
			},
			SideEffects: &noSideEffect,
		},
		}, true),
		expectedError: `clientConfig.service.path: Invalid value: "//": segment[0] may not be empty`,
	}, {
		name: "path no empty step",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "path no non-subdomain",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
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
		expectedError: `clientConfig.service.path: Invalid value: "/apis/foo.bar/v1alpha1/--bad": segment[3]: a lowercase RFC 1123 subdomain`,
	}, {
		name: "invalid port 0",
		config: newMutatingWebhookConfiguration(
			[]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "invalid port >65535",
		config: newMutatingWebhookConfiguration(
			[]admissionregistration.MutatingWebhook{{
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
	}, {
		name: "timeout seconds cannot be greater than 30",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(31),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: 31: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "timeout seconds cannot be smaller than 1",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(0),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: 0: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "timeout seconds must be positive",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &unknownSideEffect,
			TimeoutSeconds: int32Ptr(-1),
		},
		}, true),
		expectedError: `webhooks[0].timeoutSeconds: Invalid value: -1: the timeout value must be between 1 and 30 seconds`,
	}, {
		name: "valid timeout seconds",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:           "webhook.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(1),
		}, {
			Name:           "webhook2.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(15),
		}, {
			Name:           "webhook3.k8s.io",
			ClientConfig:   validClientConfig,
			SideEffects:    &noSideEffect,
			TimeoutSeconds: int32Ptr(30),
		},
		}, true),
	}, {
		name: "single match condition must have a name",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Required value`,
	}, {
		name: "all match conditions must have a name",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Expression: "true",
			}, {
				Expression: "true",
			}},
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Required value, webhooks[0].matchConditions[1].name: Required value, webhooks[1].matchConditions[0].name: Required value`,
	}, {
		name: "single match condition must have a qualified name",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "-hello",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].name: Invalid value: "-hello": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
	}, {
		name: "all match conditions must have qualified names",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       ".io",
				Expression: "true",
			}, {
				Name:       "thing.test.com",
				Expression: "true",
			}},
		}, {
			Name:         "webhook2.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "some name",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `[webhooks[0].matchConditions[0].name: Invalid value: ".io": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]'), webhooks[1].matchConditions[0].name: Invalid value: "some name": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')]`,
	}, {
		name: "expression is required",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name: "webhook.k8s.io",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Required value`,
	}, {
		name: "expression is required to have some value",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Required value`,
	}, {
		name: "invalid expression",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "object.x in [1, 2, ",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Invalid value: "object.x in [1, 2,": compilation failed: ERROR: <input>:1:19: Syntax error: missing ']' at '<EOF>'`,
	}, {
		name: "unique names same hook",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}, {
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: `matchConditions[1].name: Duplicate value: "webhook.k8s.io"`,
	}, {
		name: "repeat names allowed across different hooks",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		}, {
			Name:         "webhook2.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "true",
			}},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "must evaluate to bool",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "webhook.k8s.io",
				Expression: "6",
			}},
		},
		}, true),
		expectedError: `webhooks[0].matchConditions[0].expression: Invalid value: "6": must evaluate to bool`,
	}, {
		name: "max of 64 match conditions",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:            "webhook.k8s.io",
			ClientConfig:    validClientConfig,
			SideEffects:     &noSideEffect,
			MatchConditions: get65MatchConditions(),
		},
		}, true),
		expectedError: `webhooks[0].matchConditions: Too many: 65: must have at most 64 items`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingWebhookConfiguration(test.config)
			err := errs.ToAggregate()
			if err != nil {
				if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain:\n  %s\ngot:\n  %s", e, a)
				}
			} else {
				if test.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain:\n  %s", test.expectedError)
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
		expectedError string
	}{{
		name: "should pass on valid new AdmissionReviewVersion (v1beta1)",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should pass on valid new AdmissionReviewVersion (v1)",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"v1"},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should pass on invalid AdmissionReviewVersion with invalid previous versions",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1", "invalid-v2"},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v0"},
		},
		}, true),
		expectedError: ``,
	}, {
		name: "should fail on invalid AdmissionReviewVersion with valid previous versions",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1"},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"v1beta1", "invalid-v1"},
		},
		}, true),
		expectedError: `Invalid value: []string{"invalid-v1"}`,
	}, {
		name: "should fail on invalid AdmissionReviewVersion with missing previous versions",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:                    "webhook.k8s.io",
			ClientConfig:            validClientConfig,
			SideEffects:             &unknownSideEffect,
			AdmissionReviewVersions: []string{"invalid-v1"},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, false),
		expectedError: `Invalid value: []string{"invalid-v1"}`,
	}, {
		name: "Webhooks can have duplicate names when old config has duplicate names",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		}, {
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "Webhooks can't have side effects when old config has no side effects",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
		},
		}, true),
		expectedError: `Unsupported value: "Unknown": supported values: "None", "NoneOnDryRun"`,
	}, {
		name: "Webhooks can have side effects when old config has side effects",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &unknownSideEffect,
		},
		}, true),
		expectedError: ``,
	}, {
		name: "Webhooks must compile CEL expressions with StoredExpression environment if unchanged",
		config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "test() == true",
			}},
		},
		}, true),
		oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
			Name:         "webhook.k8s.io",
			ClientConfig: validClientConfig,
			SideEffects:  &noSideEffect,
			MatchConditions: []admissionregistration.MatchCondition{{
				Name:       "checkStorage",
				Expression: "test() == true",
			}},
		},
		}, true),
	},
		{
			name: "Webhooks must compile CEL expressions with NewExpression environment if changed",
			config: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
				Name:         "webhook.k8s.io",
				ClientConfig: validClientConfig,
				SideEffects:  &noSideEffect,
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "checkStorage",
					Expression: "test() == true",
				},
				}},
			}, true),
			oldconfig: newMutatingWebhookConfiguration([]admissionregistration.MutatingWebhook{{
				Name:         "webhook.k8s.io",
				ClientConfig: validClientConfig,
				SideEffects:  &noSideEffect,
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "checkStorage",
					Expression: "true",
				},
				}},
			}, true),
			expectedError: `undeclared reference to 'test'`,
		}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingWebhookConfigurationUpdate(test.config, test.oldconfig)
			err := errs.ToAggregate()
			if err != nil {
				if e, a := test.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain:\n  %s\ngot:\n  %s", e, a)
				}
			} else {
				if test.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain:\n  %s", test.expectedError)
				}
			}
		})

	}
}

func TestValidateValidatingAdmissionPolicy(t *testing.T) {
	tests := []struct {
		name          string
		config        *admissionregistration.ValidatingAdmissionPolicy
		expectedError string
	}{{
		name: "metadata.name validation",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!!",
			},
		},
		expectedError: `metadata.name: Invalid value: "!!!!":`,
	}, {
		name: "failure policy validation",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("other")
					return &r
				}(),
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		expectedError: `spec.failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
	}, {
		name: "failure policy validation",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("other")
					return &r
				}(),
			},
		},
		expectedError: `spec.failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
	}, {
		name: "API version is required in ParamKind",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Example",
					APIVersion: "test.example.com",
				},
			},
		},
		expectedError: `spec.paramKind.apiVersion: Invalid value: "test.example.com"`,
	}, {
		name: "API kind is required in ParamKind",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				ParamKind: &admissionregistration.ParamKind{
					APIVersion: "test.example.com/v1",
				},
			},
		},
		expectedError: `spec.paramKind.kind: Required value`,
	}, {
		name: "API version format in ParamKind",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Example",
					APIVersion: "test.example.com/!!!",
				},
			},
		},
		expectedError: `pec.paramKind.apiVersion: Invalid value: "!!!":`,
	}, {
		name: "API group format in ParamKind",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				ParamKind: &admissionregistration.ParamKind{
					APIVersion: "!!!/v1",
					Kind:       "ReplicaLimit",
				},
			},
		},
		expectedError: `pec.paramKind.apiVersion: Invalid value: "!!!":`,
	}, {
		name: "Validations is required",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{},
		},

		expectedError: `spec.validations: Required value: validations or auditAnnotations must contain at least one item`,
	}, {
		name: "Invalid Validations Reason",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
					Reason: func() *metav1.StatusReason {
						r := metav1.StatusReason("other")
						return &r
					}(),
				}},
			},
		},

		expectedError: `spec.validations[0].reason: Unsupported value: "other"`,
	}, {
		name: "MatchConstraints is required",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},

		expectedError: `spec.matchConstraints: Required value`,
	}, {
		name: "matchConstraints.resourceRules is required",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules: Required value`,
	}, {
		name: "matchConstraints.resourceRules has at least one explicit rule",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Rule: admissionregistration.Rule{},
						},
						ResourceNames: []string{"/./."},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].apiVersions: Required value`,
	}, {
		name: "expression is required",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{}},
			},
		},

		expectedError: `spec.validations[0].expression: Required value: expression is not specified`,
	}, {
		name: "matchResources resourceNames check",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						ResourceNames: []string{"/./."},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resourceNames[0]: Invalid value: "/./."`,
	}, {
		name: "matchResources resourceNames cannot duplicate",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						ResourceNames: []string{"test", "test"},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resourceNames[1]: Duplicate value: "test"`,
	}, {
		name: "matchResources validation: matchPolicy",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("other")
						return &r
					}(),
				},
			},
		},
		expectedError: `spec.matchConstraints.matchPolicy: Unsupported value: "other": supported values: "Equivalent", "Exact"`,
	}, {
		name: "Operations must not be empty or nil",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					ExcludeResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].operations: Required value, spec.matchConstraints.resourceRules[1].operations: Required value, spec.matchConstraints.excludeResourceRules[0].operations: Required value, spec.matchConstraints.excludeResourceRules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", ""},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"PATCH"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", "*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a/b", "a/*", "*/b"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
				},
			},
		},
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a/x"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "resource */a cannot mix with x/a",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/a", "x/a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "invalid expression",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x in [1, 2, ",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.validations[0].expression: Invalid value: "object.x in [1, 2, ": compilation failed: ERROR: <input>:1:20: Syntax error: missing ']' at '<EOF>`,
	}, {
		name: "invalid messageExpression",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression:        "true",
					MessageExpression: "object.x in [1, 2, ",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.validations[0].messageExpression: Invalid value: "object.x in [1, 2, ": compilation failed: ERROR: <input>:1:20: Syntax error: missing ']' at '<EOF>`,
	}, {
		name: "messageExpression of wrong type",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression:        "true",
					MessageExpression: "0 == 0",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.validations[0].messageExpression: Invalid value: "0 == 0": must evaluate to string`,
	}, {
		name: "invalid auditAnnotations key due to key name",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "@",
					ValueExpression: "value",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[0].key: Invalid value: "config/@": name part must consist of alphanumeric characters`,
	}, {
		name: "auditAnnotations keys must be unique",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "a",
					ValueExpression: "'1'",
				}, {
					Key:             "a",
					ValueExpression: "'2'",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[1].key: Duplicate value: "a"`,
	}, {
		name: "invalid auditAnnotations key due to metadata.name",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "nope!",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "key",
					ValueExpression: "'value'",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[0].key: Invalid value: "nope!/key": prefix part a lowercase RFC 1123 subdomain`,
	}, {
		name: "invalid auditAnnotations key due to length",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "this-is-a-long-name-for-a-admission-policy-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "this-is-a-long-name-for-an-audit-annotation-key-xxxxxxxxxxxxxxxxxxxxxxxxxx",
					ValueExpression: "'value'",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[0].key: Invalid value`,
	}, {
		name: "invalid auditAnnotations valueExpression type",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "something",
					ValueExpression: "true",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[0].valueExpression: Invalid value: "true": must evaluate to one of [string null_type]`,
	}, {
		name: "invalid auditAnnotations valueExpression",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				AuditAnnotations: []admissionregistration.AuditAnnotation{{
					Key:             "something",
					ValueExpression: "object.x in [1, 2, ",
				}},
			},
		},
		expectedError: `spec.auditAnnotations[0].valueExpression: Invalid value: "object.x in [1, 2, ": compilation failed: ERROR: <input>:1:19: Syntax error: missing ']' at '<EOF>`,
	}, {
		name: "single match condition must have a name",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				MatchConditions: []admissionregistration.MatchCondition{{
					Expression: "true",
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		expectedError: `spec.matchConditions[0].name: Required value`,
	}, {
		name: "match condition with parameters allowed",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Foo",
					APIVersion: "foobar/v1alpha1",
				},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		expectedError: "",
	}, {
		name: "match condition with parameters not allowed if no param kind",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		expectedError: `undeclared reference to 'params'`,
	}, {
		name: "variable composition empty name",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Variables: []admissionregistration.Variable{
					{
						Name:       "    ",
						Expression: "true",
					},
				},
				Validations: []admissionregistration.Validation{
					{
						Expression: "true",
					},
				},
			},
		},
		expectedError: `spec.variables[0].name: Required value: name is not specified`,
	}, {
		name: "variable composition name is not a valid identifier",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Variables: []admissionregistration.Variable{
					{
						Name:       "4ever",
						Expression: "true",
					},
				},
				Validations: []admissionregistration.Validation{
					{
						Expression: "true",
					},
				},
			},
		},
		expectedError: `spec.variables[0].name: Invalid value: "4ever": name is not a valid CEL identifier`,
	}, {
		name: "variable composition cannot compile",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Variables: []admissionregistration.Variable{
					{
						Name:       "foo",
						Expression: "114 + '514'", // compile error: type confusion
					},
				},
				Validations: []admissionregistration.Validation{
					{
						Expression: "true",
					},
				},
			},
		},
		expectedError: `spec.variables[0].expression: Invalid value: "114 + '514'": compilation failed: ERROR: <input>:1:5: found no matching overload for '_+_' applied to '(int, string)`,
	}, {
		name: "validation referred to non-existing variable",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Variables: []admissionregistration.Variable{
					{
						Name:       "foo",
						Expression: "1 + 1",
					},
					{
						Name:       "bar",
						Expression: "variables.foo + 1",
					},
				},
				Validations: []admissionregistration.Validation{
					{
						Expression: "variables.foo > 1", // correct
					},
					{
						Expression: "variables.replicas == 2", // replicas undefined
					},
				},
			},
		},
		expectedError: `spec.validations[1].expression: Invalid value: "variables.replicas == 2": compilation failed: ERROR: <input>:1:10: undefined field 'replicas'`,
	}, {
		name: "variables wrong order",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Variables: []admissionregistration.Variable{
					{
						Name:       "correct",
						Expression: "object",
					},
					{
						Name:       "bar", // should go below foo
						Expression: "variables.foo + 1",
					},
					{
						Name:       "foo",
						Expression: "1 + 1",
					},
				},
				Validations: []admissionregistration.Validation{
					{
						Expression: "variables.foo > 1", // correct
					},
				},
			},
		},
		expectedError: `spec.variables[1].expression: Invalid value: "variables.foo + 1": compilation failed: ERROR: <input>:1:10: undefined field 'foo'`,
	},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingAdmissionPolicy(test.config)
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

func TestValidateValidatingAdmissionPolicyUpdate(t *testing.T) {
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.ValidatingAdmissionPolicy
		config        *admissionregistration.ValidatingAdmissionPolicy
		expectedError string
	}{{
		name: "should pass on valid new ValidatingAdmissionPolicy",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "should pass on valid new ValidatingAdmissionPolicy with invalid old ValidatingAdmissionPolicy",
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{},
		},
	}, {
		name: "match conditions re-checked if paramKind changes",
		oldconfig: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Foo",
					APIVersion: "foobar/v1alpha1",
				},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		expectedError: `undeclared reference to 'params'`,
	}, {
		name: "match conditions not re-checked if no change to paramKind or matchConditions",
		oldconfig: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 100",
				}},
			},
		},
		config: &admissionregistration.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Ignore")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Validations: []admissionregistration.Validation{{
					Expression: "object.x < 50",
				}},
			},
		},
		expectedError: "",
	},
		{
			name: "expressions that are not changed must be compiled using the StoredExpression environment",
			oldconfig: validatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `test() == true`,
					},
				},
				[]admissionregistration.Validation{
					{
						Expression:        `test() == true`,
						MessageExpression: "string(test())",
					},
				},
				[]admissionregistration.AuditAnnotation{
					{
						Key:             "checkEnvironmentMode",
						ValueExpression: "string(test())",
					},
				}),
			config: validatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `test() == true`,
					},
				},
				[]admissionregistration.Validation{
					{
						Expression:        `test() == true`,
						MessageExpression: "string(test())",
					},
				},
				[]admissionregistration.AuditAnnotation{
					{
						Key:             "checkEnvironmentMode",
						ValueExpression: "string(test())",
					},
				}),
			expectedError: "",
		},
		{
			name: "matchCondition expressions that are changed must be compiled using the NewExpression environment",
			oldconfig: validatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `true`,
					},
				},
				nil, nil),
			config: validatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `test() == true`,
					},
				},
				nil, nil),
			expectedError: `undeclared reference to 'test'`,
		},
		{
			name: "validation expressions that are changed must be compiled using the NewExpression environment",
			oldconfig: validatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Validation{
					{
						Expression: `true`,
					},
				},
				nil),
			config: validatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Validation{
					{
						Expression: `test() == true`,
					},
				},
				nil),
			expectedError: `undeclared reference to 'test'`,
		},
		{
			name: "validation messageExpressions that are changed must be compiled using the NewExpression environment",
			oldconfig: validatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Validation{
					{
						Expression:        `true`,
						MessageExpression: "'test'",
					},
				},
				nil),
			config: validatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Validation{
					{
						Expression:        `true`,
						MessageExpression: "string(test())",
					},
				},
				nil),
			expectedError: `undeclared reference to 'test'`,
		},
		{
			name: "auditAnnotation valueExpressions that are changed must be compiled using the NewExpression environment",
			oldconfig: validatingAdmissionPolicyWithExpressions(
				nil, nil,
				[]admissionregistration.AuditAnnotation{
					{
						Key:             "checkEnvironmentMode",
						ValueExpression: "'test'",
					},
				}),
			config: validatingAdmissionPolicyWithExpressions(
				nil, nil,
				[]admissionregistration.AuditAnnotation{
					{
						Key:             "checkEnvironmentMode",
						ValueExpression: "string(test())",
					},
				}),
			expectedError: `undeclared reference to 'test'`,
		},
		// TODO: CustomAuditAnnotations: string valueExpression with {oldObject} is allowed
	}
	strictCost := utilfeature.DefaultFeatureGate.Enabled(features.StrictCostEnforcementForVAP)
	// Include the test library, which includes the test() function in the storage environment during test
	base := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), strictCost)
	extended, err := base.Extend(environment.VersionedOptions{
		IntroducedVersion: version.MustParseGeneric("1.999"),
		EnvOptions:        []cel.EnvOption{library.Test()},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strictCost {
		originalCompiler := getStrictStatelessCELCompiler()
		lazyStrictStatelessCELCompiler = plugincel.NewCompiler(extended)
		defer func() {
			lazyStrictStatelessCELCompiler = originalCompiler
		}()
	} else {
		originalCompiler := getNonStrictStatelessCELCompiler()
		lazyNonStrictStatelessCELCompiler = plugincel.NewCompiler(extended)
		defer func() {
			lazyNonStrictStatelessCELCompiler = originalCompiler
		}()
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingAdmissionPolicyUpdate(test.config, test.oldconfig)
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

func validatingAdmissionPolicyWithExpressions(
	matchConditions []admissionregistration.MatchCondition,
	validations []admissionregistration.Validation,
	auditAnnotations []admissionregistration.AuditAnnotation) *admissionregistration.ValidatingAdmissionPolicy {
	return &admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicySpec{
			MatchConstraints: &admissionregistration.MatchResources{
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				MatchPolicy: func() *admissionregistration.MatchPolicyType {
					r := admissionregistration.MatchPolicyType("Exact")
					return &r
				}(),
			},
			FailurePolicy: func() *admissionregistration.FailurePolicyType {
				r := admissionregistration.FailurePolicyType("Ignore")
				return &r
			}(),
			MatchConditions:  matchConditions,
			Validations:      validations,
			AuditAnnotations: auditAnnotations,
		},
	}
}

func mutatingAdmissionPolicyWithExpressions(
	matchConditions []admissionregistration.MatchCondition,
	mutations []admissionregistration.Mutation) *admissionregistration.MutatingAdmissionPolicy {
	return &admissionregistration.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "config",
		},
		Spec: admissionregistration.MutatingAdmissionPolicySpec{
			MatchConstraints: &admissionregistration.MatchResources{
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				MatchPolicy: func() *admissionregistration.MatchPolicyType {
					r := admissionregistration.MatchPolicyType("Exact")
					return &r
				}(),
			},
			FailurePolicy: func() *admissionregistration.FailurePolicyType {
				r := admissionregistration.FailurePolicyType("Ignore")
				return &r
			}(),
			MatchConditions: matchConditions,
			Mutations:       mutations,
		},
	}
}

func TestValidateValidatingAdmissionPolicyBinding(t *testing.T) {
	tests := []struct {
		name          string
		config        *admissionregistration.ValidatingAdmissionPolicyBinding
		expectedError string
	}{{
		name: "metadata.name validation",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!!",
			},
		},
		expectedError: `metadata.name: Invalid value: "!!!!":`,
	}, {
		name: "PolicyName is required",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{},
		},
		expectedError: `spec.policyName: Required value`,
	}, {
		name: "matchResources validation: matchPolicy",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("other")
						return &r
					}(),
				},
			},
		},
		expectedError: `spec.matchResouces.matchPolicy: Unsupported value: "other": supported values: "Equivalent", "Exact"`,
	}, {
		name: "Operations must not be empty or nil",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					ExcludeResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].operations: Required value, spec.matchResouces.resourceRules[1].operations: Required value, spec.matchResouces.excludeResourceRules[0].operations: Required value, spec.matchResouces.excludeResourceRules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				}, MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", ""},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				}, MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"PATCH"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", "*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a/b", "a/*", "*/b"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a/x"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "resource */a cannot mix with x/a",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/a", "x/a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "validationActions must be unique",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny, admissionregistration.Deny},
			},
		},
		expectedError: `spec.validationActions[1]: Duplicate value: "Deny"`,
	}, {
		name: "validationActions must contain supported values",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.ValidationAction("illegal")},
			},
		},
		expectedError: `Unsupported value: "illegal": supported values: "Audit", "Deny", "Warn"`,
	}, {
		name: "paramRef selector must not be set when name is set",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "xyzlimit-scale.example.com",
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				ParamRef: &admissionregistration.ParamRef{
					Name: "xyzlimit-scale-setting.example.com",
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
			},
		},
		expectedError: `spec.paramRef.name: Forbidden: name and selector are mutually exclusive, spec.paramRef.selector: Forbidden: name and selector are mutually exclusive`,
	}, {
		name: "paramRef parameterNotFoundAction must be set",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "xyzlimit-scale.example.com",
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				ParamRef: &admissionregistration.ParamRef{
					Name: "xyzlimit-scale-setting.example.com",
				},
			},
		},
		expectedError: "spec.paramRef.parameterNotFoundAction: Required value",
	}, {
		name: "paramRef parameterNotFoundAction must be an valid value",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "xyzlimit-scale.example.com",
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.ParameterNotFoundActionType("invalid")),
				},
			},
		},
		expectedError: `spec.paramRef.parameterNotFoundAction: Unsupported value: "invalid": supported values: "Deny", "Allow"`,
	}, {
		name: "paramRef one of name or selector",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "xyzlimit-scale.example.com",
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				ParamRef: &admissionregistration.ParamRef{
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
			},
		},
		expectedError: `one of name or selector must be specified`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingAdmissionPolicyBinding(test.config)
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

func TestValidateValidatingAdmissionPolicyBindingUpdate(t *testing.T) {
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.ValidatingAdmissionPolicyBinding
		config        *admissionregistration.ValidatingAdmissionPolicyBinding
		expectedError string
	}{{
		name: "should pass on valid new ValidatingAdmissionPolicyBinding",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "should pass on valid new ValidatingAdmissionPolicyBinding with invalid old ValidatingAdmissionPolicyBinding",
		config: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{},
		},
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateValidatingAdmissionPolicyBindingUpdate(test.config, test.oldconfig)
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

func TestValidateValidatingAdmissionPolicyStatus(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        *admissionregistration.ValidatingAdmissionPolicyStatus
		expectedError string
	}{{
		name:   "empty",
		status: &admissionregistration.ValidatingAdmissionPolicyStatus{},
	}, {
		name: "type checking",
		status: &admissionregistration.ValidatingAdmissionPolicyStatus{
			TypeChecking: &admissionregistration.TypeChecking{
				ExpressionWarnings: []admissionregistration.ExpressionWarning{{
					FieldRef: "spec.validations[0].expression",
					Warning:  "message",
				}},
			},
		},
	}, {
		name: "type checking bad json path",
		status: &admissionregistration.ValidatingAdmissionPolicyStatus{
			TypeChecking: &admissionregistration.TypeChecking{
				ExpressionWarnings: []admissionregistration.ExpressionWarning{{
					FieldRef: "spec[foo]",
					Warning:  "message",
				}},
			},
		},
		expectedError: "invalid JSONPath: invalid array index foo",
	}, {
		name: "type checking missing warning",
		status: &admissionregistration.ValidatingAdmissionPolicyStatus{
			TypeChecking: &admissionregistration.TypeChecking{
				ExpressionWarnings: []admissionregistration.ExpressionWarning{{
					FieldRef: "spec.validations[0].expression",
				}},
			},
		},
		expectedError: "Required value",
	}, {
		name: "type checking missing fieldRef",
		status: &admissionregistration.ValidatingAdmissionPolicyStatus{
			TypeChecking: &admissionregistration.TypeChecking{
				ExpressionWarnings: []admissionregistration.ExpressionWarning{{
					Warning: "message",
				}},
			},
		},
		expectedError: "Required value",
	},
	} {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateValidatingAdmissionPolicyStatus(tc.status, field.NewPath("status"))
			err := errs.ToAggregate()
			if err != nil {
				if e, a := tc.expectedError, err.Error(); !strings.Contains(a, e) || e == "" {
					t.Errorf("expected to contain %s, got %s", e, a)
				}
			} else {
				if tc.expectedError != "" {
					t.Errorf("unexpected no error, expected to contain %s", tc.expectedError)
				}
			}
		})
	}
}

func get65MatchConditions() []admissionregistration.MatchCondition {
	result := []admissionregistration.MatchCondition{}
	for i := 0; i < 65; i++ {
		result = append(result, admissionregistration.MatchCondition{
			Name:       fmt.Sprintf("test%v", i),
			Expression: "true",
		})
	}
	return result
}

func TestValidateMutatingAdmissionPolicy(t *testing.T) {
	applyConfigurationPatchType := admissionregistration.ApplyConfigurationPatchType
	tests := []struct {
		name          string
		config        *admissionregistration.MutatingAdmissionPolicy
		expectedError string
	}{{
		name: "metadata.name validation",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!!",
			},
		},
		expectedError: `metadata.name: Invalid value: "!!!!":`,
	}, {
		name: "failure policy validation",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("other")
					return &r
				}(),
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: `spec.failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
	}, {
		name: "patchType validation",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("other")
					return &r
				}(),
			},
		},
		expectedError: `spec.failurePolicy: Unsupported value: "other": supported values: "Fail", "Ignore"`,
	}, {
		name: "API version is required in ParamKind",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Example",
					APIVersion: "test.example.com",
				},
			},
		},
		expectedError: `spec.paramKind.apiVersion: Invalid value: "test.example.com"`,
	}, {
		name: "API kind is required in ParamKind",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				ParamKind: &admissionregistration.ParamKind{
					APIVersion: "test.example.com/v1",
				},
			},
		},
		expectedError: `spec.paramKind.kind: Required value`,
	}, {
		name: "API version format in ParamKind",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Example",
					APIVersion: "test.example.com/!!!",
				},
			},
		},
		expectedError: `pec.paramKind.apiVersion: Invalid value: "!!!":`,
	}, {
		name: "API group format in ParamKind",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				ParamKind: &admissionregistration.ParamKind{
					APIVersion: "!!!/v1",
					Kind:       "ReplicaLimit",
				},
			},
		},
		expectedError: `pec.paramKind.apiVersion: Invalid value: "!!!":`,
	}, {
		name: "Validations is required",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{},
		},

		expectedError: `spec.failurePolicy: Required value, spec.matchConstraints: Required value, spec.mutations: Required value: mutations must contain at least one item`,
	}, {
		name: "Invalid Validations Reason",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
					Reason: func() *metav1.StatusReason {
						r := metav1.StatusReason("other")
						return &r
					}(),
				}},
			},
		},

		expectedError: `spec.mutations[0].reason: Unsupported value: "other"`,
	}, {
		name: "MatchConstraints is required",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},

		expectedError: `spec.matchConstraints: Required value`,
	}, {
		name: "matchConstraints.resourceRules is required",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules: Required value`,
	}, {
		name: "matchConstraints.resourceRules has at least one explicit rule",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Rule: admissionregistration.Rule{},
						},
						ResourceNames: []string{"/./."},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].apiVersions: Required value`,
	}, {
		name: "expression is required",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{}},
			},
		},

		expectedError: `spec.mutations[0].expression: Required value: expression is not specified`,
	}, {
		name: "matchResources resourceNames check",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						ResourceNames: []string{"/./."},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resourceNames[0]: Invalid value: "/./."`,
	}, {
		name: "matchResources resourceNames cannot duplicate",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						ResourceNames: []string{"test", "test"},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resourceNames[1]: Duplicate value: "test"`,
	}, {
		name: "matchResources validation: matchPolicy",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("other")
						return &r
					}(),
				},
			},
		},
		expectedError: `spec.matchConstraints.matchPolicy: Unsupported value: "other": supported values: "Equivalent", "Exact"`,
	}, {
		name: "Operations must not be empty or nil",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					ExcludeResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].operations: Required value, spec.matchConstraints.resourceRules[1].operations: Required value, spec.matchConstraints.excludeResourceRules[0].operations: Required value, spec.matchConstraints.excludeResourceRules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", ""},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"PATCH"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", "*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a/b", "a/*", "*/b"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
				},
			},
		},
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a/x"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "resource */a cannot mix with x/a",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/a", "x/a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchConstraints.resourceRules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "invalid messageExpression",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression:        "true",
					MessageExpression: "object.x in [1, 2, ",
					PatchType:         &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.mutations[0].messageExpression: Invalid value: "object.x in [1, 2, ": compilation failed: ERROR: <input>:1:20: Syntax error: missing ']' at '<EOF>`,
	}, {
		name: "messageExpression of wrong type",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression:        "true",
					MessageExpression: "0 == 0",
					PatchType:         &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.mutations[0].messageExpression: Invalid value: "0 == 0": must evaluate to string`,
	}, {
		name: "patchType required",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "true",
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.mutations[0].patchType: Required value: patchType must be specified`,
	}, {
		name: "single match condition must have a name",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				MatchConditions: []admissionregistration.MatchCondition{{
					Expression: "true",
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "true",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: `spec.matchConditions[0].name: Required value`,
	}, {
		name: "match condition with parameters allowed",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Foo",
					APIVersion: "foobar/v1alpha1",
				},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: "",
	}, {
		name: "match condition with parameters not allowed if no param kind",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: `undeclared reference to 'params'`,
	},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingAdmissionPolicy(test.config)
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

func TestValidateMutatingAdmissionPolicyUpdate(t *testing.T) {
	applyConfigurationPatchType := admissionregistration.ApplyConfigurationPatchType
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.MutatingAdmissionPolicy
		config        *admissionregistration.MutatingAdmissionPolicy
		expectedError string
	}{{
		name: "should pass on valid new MutatingAdmissionPolicy",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "should pass on valid new MutatingAdmissionPolicy with invalid old MutatingAdmissionPolicy",
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
				MatchConstraints: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{},
		},
	}, {
		name: "match conditions re-checked if paramKind changes",
		oldconfig: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				ParamKind: &admissionregistration.ParamKind{
					Kind:       "Foo",
					APIVersion: "foobar/v1alpha1",
				},
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: `undeclared reference to 'params'`,
	}, {
		name: "match conditions not re-checked if no change to paramKind or matchConditions",
		oldconfig: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Fail")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		config: &admissionregistration.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicySpec{
				MatchConstraints: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
				},
				FailurePolicy: func() *admissionregistration.FailurePolicyType {
					r := admissionregistration.FailurePolicyType("Ignore")
					return &r
				}(),
				MatchConditions: []admissionregistration.MatchCondition{{
					Name:       "hasParams",
					Expression: `params.foo == "okay"`,
				}},
				Mutations: []admissionregistration.Mutation{{
					Expression: "object.x < 100",
					PatchType:  &applyConfigurationPatchType,
				}},
			},
		},
		expectedError: "",
	},
		{
			name: "matchCondition expressions that are changed must be compiled using the NewExpression environment",
			oldconfig: mutatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `true`,
					},
				},
				nil),
			config: mutatingAdmissionPolicyWithExpressions(
				[]admissionregistration.MatchCondition{
					{
						Name:       "checkEnvironmentMode",
						Expression: `test() == true`,
					},
				},
				nil),
			expectedError: `undeclared reference to 'test'`,
		},
		{
			name: "validation messageExpressions that are changed must be compiled using the NewExpression environment",
			oldconfig: mutatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Mutation{
					{
						Expression:        `true`,
						MessageExpression: "'test'",
						PatchType:         &applyConfigurationPatchType,
					},
				}),
			config: mutatingAdmissionPolicyWithExpressions(
				nil,
				[]admissionregistration.Mutation{
					{
						Expression:        `true`,
						MessageExpression: "string(test())",
						PatchType:         &applyConfigurationPatchType,
					},
				}),
			expectedError: `undeclared reference to 'test'`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingAdmissionPolicyUpdate(test.config, test.oldconfig)
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

func TestValidateMutatingAdmissionPolicyBinding(t *testing.T) {
	tests := []struct {
		name          string
		config        *admissionregistration.MutatingAdmissionPolicyBinding
		expectedError string
	}{{
		name: "metadata.name validation",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!!",
			},
		},
		expectedError: `metadata.name: Invalid value: "!!!!":`,
	}, {
		name: "PolicyName is required",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{},
		},
		expectedError: `spec.policyName: Required value`,
	}, {
		name: "matchResources validation: matchPolicy",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("other")
						return &r
					}(),
				},
			},
		},
		expectedError: `spec.matchResouces.matchPolicy: Unsupported value: "other": supported values: "Equivalent", "Exact"`,
	}, {
		name: "Operations must not be empty or nil",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
					ExcludeResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}, {
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: nil,
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].operations: Required value, spec.matchResouces.resourceRules[1].operations: Required value, spec.matchResouces.excludeResourceRules[0].operations: Required value, spec.matchResouces.excludeResourceRules[1].operations: Required value`,
	}, {
		name: "\"\" is NOT a valid operation",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				}, MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", ""},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: ""`,
	}, {
		name: "operation must be either create/update/delete/connect",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				}, MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"PATCH"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `Unsupported value: "PATCH"`,
	}, {
		name: "wildcard operation cannot be mixed with other strings",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE", "*"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other operations`,
	}, {
		name: `resource "*" can co-exist with resources that have subresources`,
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a/b", "a/*", "*/b"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: `resource "*" cannot mix with resources that don't have subresources`,
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `if '*' is present, must not specify other resources without subresources`,
	}, {
		name: "resource a/* cannot mix with a/x",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a/x"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources[1]: Invalid value: "a/x": if 'a/*' is present, must not specify a/x`,
	}, {
		name: "resource a/* can mix with a",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a/*", "a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "resource */a cannot mix with x/a",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/a", "x/a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources[1]: Invalid value: "x/a": if '*/a' is present, must not specify x/a`,
	}, {
		name: "resource */* cannot mix with other resources",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"*/*", "a"},
							},
						},
					}},
				},
			},
		},
		expectedError: `spec.matchResouces.resourceRules[0].resources: Invalid value: []string{"*/*", "a"}: if '*/*' is present, must not specify other resources`,
	}, {
		name: "paramRef selector must not be set when name is set",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name: "xyzlimit-scale-setting.example.com",
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
			},
		},
		expectedError: `spec.paramRef.name: Forbidden: name and selector are mutually exclusive, spec.paramRef.selector: Forbidden: name and selector are mutually exclusive`,
	}, {
		name: "paramRef parameterNotFoundAction must be set",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name: "xyzlimit-scale-setting.example.com",
				},
			},
		},
		expectedError: "spec.paramRef.parameterNotFoundAction: Required value",
	}, {
		name: "paramRef parameterNotFoundAction must be an valid value",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.ParameterNotFoundActionType("invalid")),
				},
			},
		},
		expectedError: `spec.paramRef.parameterNotFoundAction: Unsupported value: "invalid": supported values: "Deny", "Allow"`,
	}, {
		name: "paramRef one of name or selector",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
			},
		},
		expectedError: `one of name or selector must be specified`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingAdmissionPolicyBinding(test.config)
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

func TestValidateMutatingAdmissionPolicyBindingUpdate(t *testing.T) {
	tests := []struct {
		name          string
		oldconfig     *admissionregistration.MutatingAdmissionPolicyBinding
		config        *admissionregistration.MutatingAdmissionPolicyBinding
		expectedError string
	}{{
		name: "should pass on valid new MutatingAdmissionPolicyBinding",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
	}, {
		name: "should pass on valid new MutatingAdmissionPolicyBinding with invalid old ValidatingAdmissionPolicyBinding",
		config: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "config",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "xyzlimit-scale.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "xyzlimit-scale-setting.example.com",
					ParameterNotFoundAction: ptr(admissionregistration.DenyAction),
				},
				MatchResources: &admissionregistration.MatchResources{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					ObjectSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					MatchPolicy: func() *admissionregistration.MatchPolicyType {
						r := admissionregistration.MatchPolicyType("Exact")
						return &r
					}(),
					ResourceRules: []admissionregistration.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					}},
				},
			},
		},
		oldconfig: &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "!!!",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{},
		},
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateMutatingAdmissionPolicyBindingUpdate(test.config, test.oldconfig)
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
