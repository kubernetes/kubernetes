/*
Copyright 2022 The Kubernetes Authors.

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
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/require"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	v1 "k8s.io/api/core/v1"
	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestMutatingAdmissionPolicy tests MutatingAdmissionPolicy using a shared apiserver for all tests
// and waiting for bindings to become ready by dry-running marker requests until the binding successfully
// mutates a marker, and then verifies the policy exactly once.
func TestMutatingAdmissionPolicy(t *testing.T) {
	allow := admissionregistrationv1.AllowAction
	matchEndpointResources := admissionregistrationv1.MatchResources{
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{"*"},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"endpoints"},
					},
				},
			},
		},
	}

	cases := []struct {
		name     string
		policies []*admissionregistrationv1.MutatingAdmissionPolicy
		bindings []*admissionregistrationv1.MutatingAdmissionPolicyBinding
		params   []*v1.ConfigMap

		requestOperation admissionregistrationv1.OperationType
		requestResource  schema.GroupVersionResource
		subresources     []string // Only supported for requestOperation=Update since subresources can not be created
		requestObject    runtime.Object
		expected         runtime.Object
	}{
		{
			name: "basic",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("basic-policy", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"my-foo-annotation": "myAnnotationValue"
								}
							}
						}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("basic-policy", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-policy-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-policy-object",
					Namespace: "default",
					Annotations: map[string]string{
						"my-foo-annotation": "myAnnotationValue",
					},
				},
			},
		},
		{
			name: "multiple policies",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-policy-1", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo1": "foo1Value"
								}
							}
						}`,
					},
				}),
				mutatingPolicy("multi-policy-2", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo2": "foo2Value"
								}
							}
						}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-policy-1", nil, nil),
				mutatingBinding("multi-policy-2", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-policy-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-policy-object",
					Namespace: "default",
					Annotations: map[string]string{
						"foo1": "foo1Value",
						"foo2": "foo2Value",
					},
				},
			},
		},
		{
			name: "policy with native param",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("policy-with-native", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, &admissionregistrationv1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									params.data["key"]: params.data["value"]
								}
							}
						}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-with-native", &admissionregistrationv1.ParamRef{
					Name:                    "policy-with-native-param",
					ParameterNotFoundAction: &allow,
				}, nil),
			},
			params: []*v1.ConfigMap{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "policy-with-native-param",
						Namespace: "default",
					},
					Data: map[string]string{
						"key":   "myFooKey",
						"value": "myFooValue",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-native-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-native-object",
					Namespace: "default",
					Annotations: map[string]string{
						"myFooKey": "myFooValue",
					},
				},
			},
		},
		{
			name: "policy with multiple params quantified by single binding",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-param-binding", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, &admissionregistrationv1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-param-binding", &admissionregistrationv1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"multi-param-binding-param": "true",
						},
					},
					Namespace:               "default",
					ParameterNotFoundAction: &allow,
				}, nil),
			},
			params: []*v1.ConfigMap{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-param-binding-param-1",
						Namespace: "default",
						Labels: map[string]string{
							"multi-param-binding-param": "true",
						},
					},
					Data: map[string]string{
						"multi-param-binding-key1": "value1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-param-binding-param-2",
						Namespace: "default",
						Labels: map[string]string{
							"multi-param-binding-param": "true",
						},
					},
					Data: map[string]string{
						"multi-param-binding-key2": "value2",
						"multi-param-binding-key3": "value3",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-binding-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-binding-object",
					Namespace: "default",
					Annotations: map[string]string{
						"multi-param-binding-key1": "value1",
						"multi-param-binding-key2": "value2",
						"multi-param-binding-key3": "value3",
					},
				},
			},
		},
		{
			name: "policy with variables",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				withMutatingVariables([]admissionregistrationv1.Variable{
					{Name: "foo1", Expression: `"foo1" + "Value"`},
					{Name: "foo2", Expression: `variables.foo1.replace("1", "2")`},
				},
					mutatingPolicy("policy-with-multiple-mutations", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil,
						admissionregistrationv1.Mutation{
							PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
							ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
								Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"foo1": variables.foo1
									}
								}
							}`,
							},
						},
						admissionregistrationv1.Mutation{
							PatchType: admissionregistrationv1.PatchTypeJSONPatch,
							JSONPatch: &admissionregistrationv1.JSONPatch{
								Expression: `[
									JSONPatch{op: "test", path: "/metadata/annotations", value: {"foo1": variables.foo1}},
									JSONPatch{op: "add", path: "/metadata/annotations/foo2", value: variables.foo2},
								]`,
							},
						},
						admissionregistrationv1.Mutation{
							PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
							ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
								Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"foo3": "foo3Value"
									}
								}
							}`,
							},
						},
					)),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-with-multiple-mutations", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-multiple-mutations-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-multiple-mutations-object",
					Namespace: "default",
					Annotations: map[string]string{
						"foo1": "foo1Value",
						"foo2": "foo2Value",
						"foo3": "foo3Value",
					},
				},
			},
		},
		{
			name: "match condition matches",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]admissionregistrationv1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-match-condition", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-match-condition", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test"},
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test", "applied": "updated"},
				},
			},
		},
		{
			// same as the multiple mutations test, but the mutations are split
			// across multiple policies
			name: "multiple policies requiring reinvocation",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("policy-1", admissionregistrationv1.IfNeededReinvocationPolicy, matchEndpointResources, nil,
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
										"firstApplied": "true"
									}
								}
							}`,
						},
					},
				),
				mutatingPolicy("policy-2", admissionregistrationv1.IfNeededReinvocationPolicy, matchEndpointResources, nil,
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
										"secondApplied": "true"
									}
								}
							}`,
						},
					},
				),
				mutatingPolicy("policy-3", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil,
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
										"thirdApplied": "true"
									}
								}
							}`,
						},
					},
				),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-1", nil, nil),
				mutatingBinding("policy-2", nil, nil),
				mutatingBinding("policy-3", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						// First mutation 0->1
						// Second mutation 1->2
						// Third Mutation 2->3
						// First Mutation Reinvocation 3->4
						// Second Mutation Reinvocation 4->5
						// (Third Mutation is set to never reinvocation, so it's not reinvoked)
						// No future reinvocation passes (we only do a single reinvocation)
						"foo":           "5",
						"firstApplied":  "true",
						"secondApplied": "true",
						"thirdApplied":  "true",
					},
				},
			},
		},
	}

	// Run all tests in a shared apiserver
	versions := []schema.GroupVersion{admissionregistrationv1.SchemeGroupVersion, admissionregistrationv1beta1.SchemeGroupVersion}
	for _, version := range versions {
		t.Run(version.Version, func(t *testing.T) {
			var flags []string
			if version == admissionregistrationv1beta1.SchemeGroupVersion {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
				flags = []string{fmt.Sprintf("--runtime-config=%s=true", version)}
			}
			server, err := apiservertesting.StartTestServer(t, nil, flags, framework.SharedEtcd())
			require.NoError(t, err)
			defer server.TearDownFn()

			client, err := kubernetes.NewForConfig(server.ClientConfig)
			require.NoError(t, err)

			dynClient, err := dynamic.NewForConfig(server.ClientConfig)
			require.NoError(t, err)

			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			for i, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {

					// Create the policies, bindings and params.
					for _, param := range tc.params {
						_, err = client.CoreV1().ConfigMaps(param.GetNamespace()).Create(ctx, param, metav1.CreateOptions{FieldManager: "integration-test"})
						require.NoError(t, err)
					}

					for _, p := range tc.policies {
						// Modify each policy to also mutate marker requests.
						p = withMutatingWaitReadyConstraintAndExpression(p, fmt.Sprintf("%d-%s", i, p.Name))
						err = createPolicy(ctx, t, client, version, p)
						require.NoError(t, err)
					}

					for _, b := range tc.bindings {
						// After creating each binding, wait until a marker request is successfully mutated.
						err = createAndWaitReadyMutating(ctx, t, client, version, b, fmt.Sprintf("%d-%s", i, b.Spec.PolicyName))
						require.NoError(t, err)
					}

					unstructuredRequestObj := toUnstructured(t, tc.requestObject)
					unstructuredExpectedObj := toUnstructured(t, tc.expected)
					wipeUncheckedFields(t, unstructuredExpectedObj)

					defer func() {
						if cleanupErr := cleanupMutatingPolicy(ctx, t, dynClient, version, tc.policies, tc.bindings, tc.params); cleanupErr != nil {
							t.Logf("error while cleaning up policy and its bindings: %v", cleanupErr)
						}
					}()

					// Verify that the policy is working as expected.
					// Note that we do NOT retry requests here. Once the bindings are verified as working via marker
					// requests, we expect the policy to work consistently for all subsequent requests.
					var resultObj runtime.Object
					rsrcClient := clientForType(t, unstructuredRequestObj, tc.requestResource, dynClient)
					switch tc.requestOperation {
					case admissionregistrationv1.Create:
						resultObj, err = rsrcClient.Create(ctx, unstructuredRequestObj, metav1.CreateOptions{
							DryRun:       []string{metav1.DryRunAll},
							FieldManager: "integration-test",
						}, tc.subresources...)
					case admissionregistrationv1.Update:
						resultObj, err = rsrcClient.Update(ctx, unstructuredRequestObj, metav1.UpdateOptions{
							DryRun:       []string{metav1.DryRunAll},
							FieldManager: "integration-test",
						}, tc.subresources...)
						require.NoError(t, err)
					default:
						t.Fatalf("unsupported operation: %v", tc.requestOperation)
					}
					wipeUncheckedFields(t, resultObj)
					if !cmp.Equal(unstructuredExpectedObj, resultObj, cmpopts.EquateEmpty()) {
						t.Errorf("unexpected diff:\n%s\n", cmp.Diff(unstructuredRequestObj, resultObj, cmpopts.EquateEmpty()))
					}
				})
			}
		})
	}
}

// TestMutatingAdmissionPolicy_Slow tests policies by waiting until a request is successfully mutated.
// This is slower because it creates an apiserver for each whereas TestMutatingAdmissionPolicy creates
// a single apiserver and then uses marker requests to check that a binding is ready before testing it exactly once.
// Only test cases that cannot be run in TestMutatingAdmissionPolicy should be added here.
func TestMutatingAdmissionPolicy_Slow(t *testing.T) {
	allow := admissionregistrationv1.AllowAction
	matchEndpointResources := admissionregistrationv1.MatchResources{
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{"*"},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"endpoints"},
					},
				},
			},
		},
	}

	cases := []struct {
		name     string
		policies []*admissionregistrationv1.MutatingAdmissionPolicy
		bindings []*admissionregistrationv1.MutatingAdmissionPolicyBinding
		params   []*v1.ConfigMap

		requestOperation admissionregistrationv1.OperationType
		requestResource  schema.GroupVersionResource
		subresources     []string       // Only supported for requestOperation=Update since subresources can not be created
		initialObject    runtime.Object // For requestOperation=Update, this may be used to create the initial object state
		requestObject    runtime.Object
		expected         runtime.Object
	}{
		{
			name: "unbound policy is no-op",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("unbound-policy", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo": "fooValue"
								}
							}
						}`,
					},
				}),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
		},
		{
			name: "failure policy ignore",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				withMutatingFailurePolicy(admissionregistrationv1.Ignore,
					mutatingPolicy("policy", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{spec: Object.spec{invalidField: "invalid apply configuration"}}`,
						},
					})),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
		},
		{
			name: "match condition does not match",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]admissionregistrationv1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-no-match-condition", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-no-match-condition", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "production"},
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "production"},
				},
			},
		},
		{
			name: "some policy conditions match",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]admissionregistrationv1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("production")`}},
					mutatingPolicy("policy-1", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "wrong"}}}`,
						},
					})),
				withMutatingMatchConditions([]admissionregistrationv1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-2", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, nil, admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-1", nil, nil),
				mutatingBinding("policy-2", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test"},
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test", "applied": "updated"},
				},
			},
		},
		{
			name: "mutate status subresource",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("subresource-status", admissionregistrationv1.NeverReinvocationPolicy, admissionregistrationv1.MatchResources{
					ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
						{
							RuleWithOperations: admissionregistrationv1.RuleWithOperations{
								Operations: []admissionregistrationv1.OperationType{"*"},
								Rule: admissionregistrationv1.Rule{
									APIGroups:   []string{""},
									APIVersions: []string{"v1"},
									Resources:   []string{"namespaces/status"},
								},
							},
						},
					},
				}, nil, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `Object{
							status: Object.status{
								conditions: [Object.status.conditions{
									type: "NamespaceDeletionContentFailure", 
									message: "mutated"
								}]
							}
						}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("subresource-status", nil, nil),
			},
			requestOperation: admissionregistrationv1.Update,
			subresources:     []string{"status"},
			requestResource:  v1.SchemeGroupVersion.WithResource("namespaces"),
			initialObject: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-namespace",
				},
			},
			requestObject: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-namespace",
				},
				Status: v1.NamespaceStatus{
					Conditions: []v1.NamespaceCondition{{
						Type:   v1.NamespaceDeletionContentFailure,
						Status: v1.ConditionUnknown,
					}},
				},
			},
			expected: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test-namespace",
					Labels: map[string]string{"kubernetes.io/metadata.name": "test-namespace"},
				},
				Spec: v1.NamespaceSpec{
					Finalizers: []v1.FinalizerName{"kubernetes"},
				},
				Status: v1.NamespaceStatus{
					Conditions: []v1.NamespaceCondition{{
						Type:    v1.NamespaceDeletionContentFailure,
						Status:  v1.ConditionUnknown,
						Message: "mutated",
					}},
					Phase: v1.NamespaceActive,
				},
			},
		},
		{
			name: "multiple bindings with different params",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-binding", admissionregistrationv1.NeverReinvocationPolicy, matchEndpointResources, &admissionregistrationv1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
				}),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-binding", &admissionregistrationv1.ParamRef{
					Name:                    "multi-binding-param-1",
					Namespace:               "default",
					ParameterNotFoundAction: &allow,
				}, nil),
				mutatingBinding("multi-binding", &admissionregistrationv1.ParamRef{
					Name:                    "multi-binding-param-2",
					Namespace:               "default",
					ParameterNotFoundAction: &allow,
				}, nil),
			},
			params: []*v1.ConfigMap{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-binding-param-1",
						Namespace: "default",
					},
					Data: map[string]string{
						"multi-binding-key1": "value1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-binding-param-2",
						Namespace: "default",
					},
					Data: map[string]string{
						"multi-binding-key2": "value2",
						"multi-binding-key3": "value3",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-binding-object",
					Namespace: "default",
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-binding-object",
					Namespace: "default",
					Annotations: map[string]string{
						"multi-binding-key1": "value1",
						"multi-binding-key2": "value2",
						"multi-binding-key3": "value3",
					},
				},
			},
		},
		{
			// Same as the other cases, but the reinvocation is caused by
			// multiple params bound
			name: "multiple params causing reinvocation",
			policies: []*admissionregistrationv1.MutatingAdmissionPolicy{
				mutatingPolicy(
					"multi-param-reinvocation",
					admissionregistrationv1.IfNeededReinvocationPolicy,
					matchEndpointResources,
					&admissionregistrationv1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
						},
					},
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
									}
								}
							}`,
						},
					},
				),
				mutatingPolicy(
					"policy-with-param-no-reinvoke",
					admissionregistrationv1.NeverReinvocationPolicy,
					matchEndpointResources,
					&admissionregistrationv1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
						},
					},
					admissionregistrationv1.Mutation{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
									}
								}
							}`,
						},
					},
				),
			},
			bindings: []*admissionregistrationv1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-param-reinvocation", &admissionregistrationv1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"multi-param-reinvocation": "true",
						},
					},
					ParameterNotFoundAction: &allow,
				}, nil),
				mutatingBinding("policy-with-param-no-reinvoke", &admissionregistrationv1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Name:                    "multi-param-reinvocation-param-3",
					ParameterNotFoundAction: &allow,
				}, nil),
			},
			params: []*v1.ConfigMap{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-param-reinvocation-param-1",
						Namespace: "default",
						Labels: map[string]string{
							"multi-param-reinvocation": "true",
						},
					},
					Data: map[string]string{
						"firstApplied": "true",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-param-reinvocation-param-2",
						Namespace: "default",
						Labels: map[string]string{
							"multi-param-reinvocation": "true",
						},
					},
					Data: map[string]string{
						"secondApplied": "true",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "multi-param-reinvocation-param-3",
						Namespace: "default",
					},
					Data: map[string]string{
						"thirdApplied": "true",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  v1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-reinvocation-object",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-reinvocation-object",
					Namespace: "default",
					Annotations: map[string]string{
						"firstApplied":  "true",
						"secondApplied": "true",
						"thirdApplied":  "true",
						"foo":           "5",
					},
				},
			},
		},
	}
	versions := []schema.GroupVersion{admissionregistrationv1.SchemeGroupVersion, admissionregistrationv1beta1.SchemeGroupVersion}
	for _, version := range versions {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%s", version.Version, tc.name), func(t *testing.T) {
				var flags []string
				if version == admissionregistrationv1beta1.SchemeGroupVersion {
					flags = []string{fmt.Sprintf("--runtime-config=%s=true", version)}
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
				}
				server, err := apiservertesting.StartTestServer(t, nil, flags, framework.SharedEtcd())
				require.NoError(t, err)
				defer server.TearDownFn()

				client, err := kubernetes.NewForConfig(server.ClientConfig)
				require.NoError(t, err)

				dynClient, err := dynamic.NewForConfig(server.ClientConfig)
				require.NoError(t, err)

				ctx, cancel := context.WithCancel(context.Background())
				t.Cleanup(cancel)

				for _, param := range tc.params {
					_, err = client.CoreV1().ConfigMaps(param.GetNamespace()).Create(ctx, param, metav1.CreateOptions{FieldManager: "integration-test"})
					require.NoError(t, err)
				}

				for _, p := range tc.policies {
					err = createPolicy(ctx, t, client, version, p)
					require.NoError(t, err)
				}

				for _, b := range tc.bindings {
					err = createBinding(ctx, t, client, version, b)
					require.NoError(t, err)
				}

				if tc.initialObject != nil {
					initClient := clientForType(t, tc.initialObject, tc.requestResource, dynClient)
					_, err = initClient.Create(ctx, toUnstructured(t, tc.initialObject), metav1.CreateOptions{})
					require.NoError(t, err)
				}

				unstructuredRequestObj := toUnstructured(t, tc.requestObject)
				unstructuredExpectedObj := toUnstructured(t, tc.expected)
				wipeUncheckedFields(t, unstructuredExpectedObj)

				// Dry Run the request until we get the expected mutated response
				var resultObj runtime.Object
				err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (done bool, err error) {
					rsrcClient := clientForType(t, unstructuredRequestObj, tc.requestResource, dynClient)
					switch tc.requestOperation {
					case admissionregistrationv1.Create:
						resultObj, err = rsrcClient.Create(ctx, unstructuredRequestObj, metav1.CreateOptions{
							DryRun:       []string{metav1.DryRunAll},
							FieldManager: "integration-test",
						}, tc.subresources...)
						if err != nil {
							t.Logf("error while waiting: %v", err)
							return false, nil
						}
						wipeUncheckedFields(t, resultObj)
						return reflect.DeepEqual(unstructuredExpectedObj, resultObj), nil
					case admissionregistrationv1.Update:
						resultObj, err = rsrcClient.Update(ctx, unstructuredRequestObj, metav1.UpdateOptions{
							DryRun:       []string{metav1.DryRunAll},
							FieldManager: "integration-test",
						}, tc.subresources...)
						if err != nil {
							t.Logf("error while waiting: %v", err)
							return false, nil
						}
						wipeUncheckedFields(t, resultObj)
						return reflect.DeepEqual(unstructuredExpectedObj, resultObj), nil
					default:
						t.Fatalf("unsupported operation: %v", tc.requestOperation)
					}
					return false, nil
				})

				if errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("failed to get expected result before timeout: %v", cmp.Diff(unstructuredExpectedObj, resultObj))
				} else if err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// Test_MutatingAdmissionPolicy_CustomResources tests a custom resource mutation.
// CRDs are also ideal for testing version conversion since old version are not removed, so version conversion is also
// tested.
func Test_MutatingAdmissionPolicy_CustomResources(t *testing.T) {
	versions := []schema.GroupVersion{admissionregistrationv1.SchemeGroupVersion, admissionregistrationv1beta1.SchemeGroupVersion}
	for _, version := range versions {
		t.Run(version.Version, func(t *testing.T) {
			var flags []string
			if version == admissionregistrationv1beta1.SchemeGroupVersion {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
				flags = []string{fmt.Sprintf("--runtime-config=%s=true", version)}
			}
			server, err := apiservertesting.StartTestServer(t, nil, flags, framework.SharedEtcd())
			etcd.CreateTestCRDs(t, apiextensions.NewForConfigOrDie(server.ClientConfig), false, versionedCustomResourceDefinition())
			if err != nil {
				t.Fatal(err)
			}
			defer server.TearDownFn()

			config := server.ClientConfig

			client, err := kubernetes.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			policy := withMutatingFailurePolicy(admissionregistrationv1.Fail, mutatingPolicy("match-by-match-policy-equivalent", admissionregistrationv1.IfNeededReinvocationPolicy, admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistrationv1.RuleWithOperations{
							Operations: []admissionregistrationv1.OperationType{
								"*",
							},
							Rule: admissionregistrationv1.Rule{
								APIGroups: []string{
									"awesome.bears.com",
								},
								APIVersions: []string{
									"v1",
								},
								Resources: []string{
									"pandas",
								},
							},
						},
					},
				}},
				nil,
				admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `Object{ metadata: Object.metadata{ labels: {"mutated-panda": "true"} } }`,
					},
				},
			))
			testID := "policy-equivalent"
			policy = withMutatingWaitReadyConstraintAndExpression(policy, testID)
			if err := createPolicy(ctx, t, client, version, policy); err != nil {
				t.Fatal(err)
			}

			policyBinding := mutatingBinding("match-by-match-policy-equivalent", nil, nil)
			if err := createAndWaitReadyMutating(ctx, t, client, version, policyBinding, testID); err != nil {
				t.Fatal(err)
			}

			v1Resource := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "awesome.bears.com" + "/" + "v1",
					"kind":       "Panda",
					"metadata": map[string]interface{}{
						"name": "v1-bears",
					},
				},
			}

			v2Resource := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "awesome.bears.com" + "/" + "v2",
					"kind":       "Panda",
					"metadata": map[string]interface{}{
						"name": "v2-bears",
					},
				},
			}

			dynamicClient, err := dynamic.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}

			// Wait for CRDs to register
			err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				createdv1, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v1", Resource: "pandas"}).Create(ctx, v1Resource, metav1.CreateOptions{})
				if err != nil {
					if strings.Contains(err.Error(), "Resource kind awesome.bears.com/v1, Kind=Panda not found") {
						return false, nil
					}
					return false, nil
				}
				if createdv1.GetLabels()["mutated-panda"] != "true" {
					t.Errorf("expected mutated-panda to be true, got %s", createdv1.GetLabels())
				}
				return true, nil
			})
			if err != nil {
				t.Fatal(err)
			}

			err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				createdv2, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v2", Resource: "pandas"}).Create(ctx, v2Resource, metav1.CreateOptions{})
				if err != nil {
					if strings.Contains(err.Error(), "Resource kind awesome.bears.com/v2, Kind=Panda not found") {
						return false, nil
					}
					return false, nil
				}
				if createdv2.GetLabels()["mutated-panda"] != "true" {
					t.Errorf("expected mutated-panda to be true, got %s", createdv2.GetLabels())
				}
				return true, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func mutatingPolicy(name string, reinvocationPolicy admissionregistrationv1.ReinvocationPolicyType, matchResources admissionregistrationv1.MatchResources, paramKind *admissionregistrationv1.ParamKind, mutations ...admissionregistrationv1.Mutation) *admissionregistrationv1.MutatingAdmissionPolicy {
	return &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			MatchConstraints:   &matchResources,
			ParamKind:          paramKind,
			Mutations:          mutations,
			ReinvocationPolicy: reinvocationPolicy,
		},
	}
}

func withMutatingVariables(variables []admissionregistrationv1.Variable, policy *admissionregistrationv1.MutatingAdmissionPolicy) *admissionregistrationv1.MutatingAdmissionPolicy {
	policy.Spec.Variables = variables
	return policy
}

func withMutatingFailurePolicy(failure admissionregistrationv1.FailurePolicyType, policy *admissionregistrationv1.MutatingAdmissionPolicy) *admissionregistrationv1.MutatingAdmissionPolicy {
	policy.Spec.FailurePolicy = &failure
	return policy
}

func withMutatingMatchConditions(matchConditions []admissionregistrationv1.MatchCondition, policy *admissionregistrationv1.MutatingAdmissionPolicy) *admissionregistrationv1.MutatingAdmissionPolicy {
	policy.Spec.MatchConditions = matchConditions
	return policy
}

func withMutatingWaitReadyConstraintAndExpression(policy *admissionregistrationv1.MutatingAdmissionPolicy, testID string) *admissionregistrationv1.MutatingAdmissionPolicy {
	policy = policy.DeepCopy()
	policy.Spec.MatchConstraints.ResourceRules = append(policy.Spec.MatchConstraints.ResourceRules, admissionregistrationv1.NamedRuleWithOperations{
		ResourceNames: []string{"test-marker"},
		RuleWithOperations: admissionregistrationv1.RuleWithOperations{
			Operations: []admissionregistrationv1.OperationType{
				"CREATE",
			},
			Rule: admissionregistrationv1.Rule{
				APIGroups: []string{
					"",
				},
				APIVersions: []string{
					"v1",
				},
				Resources: []string{
					"endpoints",
				},
			},
		},
	})
	for i, mc := range policy.Spec.MatchConditions {
		mc.Expression = `object.metadata.?labels["mutation-marker"].hasValue() || ` + mc.Expression
		policy.Spec.MatchConditions[i] = mc
	}
	for _, m := range policy.Spec.Mutations {
		if m.ApplyConfiguration != nil {
			bypass := `object.metadata.?labels["mutation-marker"].hasValue() ? Object{} : `
			m.ApplyConfiguration.Expression = bypass + m.ApplyConfiguration.Expression
		}
	}
	policy.Spec.Mutations = append([]admissionregistrationv1.Mutation{{
		PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
		ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
			// Only mutate mutation-markers.
			Expression: fmt.Sprintf(`object.metadata.?labels["mutation-marker"] == optional.of("%v") ? Object{ metadata: Object.metadata{ labels: {"mutated":"%v"}}}: Object{}`, testID, testID),
		},
	}}, policy.Spec.Mutations...)
	return policy
}

func mutatingBinding(policyName string, paramRef *admissionregistrationv1.ParamRef, matchResources *admissionregistrationv1.MatchResources) *admissionregistrationv1.MutatingAdmissionPolicyBinding {
	return &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: policyName + "-binding-" + string(uuid.NewUUID()),
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName:     policyName,
			ParamRef:       paramRef,
			MatchResources: matchResources,
		},
	}
}

func createAndWaitReadyMutating(ctx context.Context, t *testing.T, client kubernetes.Interface, version schema.GroupVersion, binding *admissionregistrationv1.MutatingAdmissionPolicyBinding, testID string) error {
	return createAndWaitReadyNamespacedMutating(ctx, t, client, version, binding, "default", testID)
}

func createAndWaitReadyNamespacedMutating(ctx context.Context, t *testing.T, client kubernetes.Interface, version schema.GroupVersion, binding *admissionregistrationv1.MutatingAdmissionPolicyBinding, ns string, testID string) error {
	if err := createBinding(ctx, t, client, version, binding); err != nil {
		return err
	}
	return waitReadyMutating(ctx, t, client, ns, testID)
}

func waitReadyMutating(ctx context.Context, t *testing.T, client kubernetes.Interface, ns string, testID string) error {
	marker := &v1.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "test-marker", Namespace: ns, Labels: map[string]string{"mutation-marker": testID}}}
	if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		result, err := client.CoreV1().Endpoints(ns).Create(ctx, marker, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}, FieldManager: "mutation-marker-sender"})
		if err != nil {
			if strings.Contains(err.Error(), "no params found for policy binding") { // wait for params to register
				return false, nil
			} else {
				return false, err
			}
		}
		if result.Labels["mutated"] == testID {
			return true, nil
		}
		return false, nil
	}); waitErr != nil {
		return waitErr
	}
	t.Logf("Marker ready: %v", marker)
	return nil
}

func cleanupMutatingPolicy(ctx context.Context, t *testing.T, client dynamic.Interface, version schema.GroupVersion, policies []*admissionregistrationv1.MutatingAdmissionPolicy, bindings []*admissionregistrationv1.MutatingAdmissionPolicyBinding, params []*v1.ConfigMap) error {
	policyGVR := version.WithResource("mutatingadmissionpolicies")
	bindingGVR := version.WithResource("mutatingadmissionpolicybindings")
	cmGVR := v1.SchemeGroupVersion.WithResource("configmaps")

	for _, policy := range policies {
		if err := client.Resource(policyGVR).Delete(ctx, policy.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
			_, err := client.Resource(policyGVR).Get(ctx, policy.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}); waitErr != nil {
			t.Fatalf("timed out waiting for policy to be cleaned up: %v", waitErr)
		}
	}

	for _, binding := range bindings {
		err := client.Resource(bindingGVR).Delete(ctx, binding.Name, metav1.DeleteOptions{})
		if err != nil {
			t.Fatal(err)
		}

		if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
			_, err := client.Resource(bindingGVR).Get(ctx, binding.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}); waitErr != nil {
			t.Fatalf("timed out waiting for policy binding to be cleaned up: %v", err)
		}
	}

	for _, param := range params {
		if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
			if err := client.Resource(cmGVR).Namespace(param.GetNamespace()).Delete(ctx, param.Name, metav1.DeleteOptions{}); err != nil {
				if apierrors.IsNotFound(err) {
					return true, nil
				}
				return false, nil
			}
			_, err := client.Resource(cmGVR).Namespace(param.GetNamespace()).Get(ctx, param.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}); waitErr != nil {
			t.Fatalf("timed out waiting for policy to be cleaned up: %v", waitErr)
		}
	}

	return nil
}

func clientForType(t *testing.T, obj runtime.Object, resource schema.GroupVersionResource, dynClient *dynamic.DynamicClient) dynamic.ResourceInterface {
	acc, err := meta.Accessor(obj)
	require.NoError(t, err)
	var rsrcClient dynamic.ResourceInterface = dynClient.Resource(resource)
	if len(acc.GetNamespace()) > 0 {
		rsrcClient = rsrcClient.(dynamic.NamespaceableResourceInterface).Namespace(acc.GetNamespace())
	}
	return rsrcClient
}

func toUnstructured(t *testing.T, obj runtime.Object) *unstructured.Unstructured {
	unstructuredRequestMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	require.NoError(t, err)

	unstructuredRequestObj := &unstructured.Unstructured{
		Object: unstructuredRequestMap,
	}
	return unstructuredRequestObj
}

func wipeUncheckedFields(t *testing.T, obj runtime.Object) {
	acc, err := meta.Accessor(obj)
	require.NoError(t, err)

	// GVK can't be patched, and not always on our test objects, so
	// clear for convenience
	obj.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})

	// Will be set by server, should be wiped
	acc.SetResourceVersion("")
	acc.SetUID("")
	acc.SetCreationTimestamp(metav1.Time{})
	acc.SetManagedFields(nil)
}

func convertObject[T any](t *testing.T, source interface{}) *T {
	data, err := json.Marshal(source)
	require.NoError(t, err)
	var dest T
	err = json.Unmarshal(data, &dest)
	require.NoError(t, err)
	return &dest
}

func createPolicy(ctx context.Context, t *testing.T, client kubernetes.Interface, version schema.GroupVersion, policy *admissionregistrationv1.MutatingAdmissionPolicy) error {
	if version == admissionregistrationv1beta1.SchemeGroupVersion {
		v1beta1P := convertObject[admissionregistrationv1beta1.MutatingAdmissionPolicy](t, policy)
		_, err := client.AdmissionregistrationV1beta1().MutatingAdmissionPolicies().Create(ctx, v1beta1P, metav1.CreateOptions{FieldManager: "integration-test"})
		return err
	}
	_, err := client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{FieldManager: "integration-test"})
	return err
}

func createBinding(ctx context.Context, t *testing.T, client kubernetes.Interface, version schema.GroupVersion, binding *admissionregistrationv1.MutatingAdmissionPolicyBinding) error {
	if version == admissionregistrationv1beta1.SchemeGroupVersion {
		v1beta1Binding := convertObject[admissionregistrationv1beta1.MutatingAdmissionPolicyBinding](t, binding)
		_, err := client.AdmissionregistrationV1beta1().MutatingAdmissionPolicyBindings().Create(ctx, v1beta1Binding, metav1.CreateOptions{FieldManager: "integration-test"})
		return err
	}
	_, err := client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{FieldManager: "integration-test"})
	return err
}
