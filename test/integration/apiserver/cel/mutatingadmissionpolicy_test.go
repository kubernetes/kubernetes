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
	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
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
	matchEndpointResources := v1alpha1.MatchResources{
		ResourceRules: []v1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: v1alpha1.RuleWithOperations{
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
		policies []*v1alpha1.MutatingAdmissionPolicy
		bindings []*v1alpha1.MutatingAdmissionPolicyBinding
		params   []*corev1.ConfigMap

		requestOperation admissionregistrationv1.OperationType
		requestResource  schema.GroupVersionResource
		subresources     []string // Only supported for requestOperation=Update since subresources can not be created
		requestObject    runtime.Object
		expected         runtime.Object
	}{
		{
			name: "basic",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("basic-policy", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("basic-policy", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-policy-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-policy-1", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
				mutatingPolicy("multi-policy-2", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-policy-1", nil, nil),
				mutatingBinding("multi-policy-2", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-policy-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("policy-with-native", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-with-native", &v1alpha1.ParamRef{
					Name: "policy-with-native-param",
				}, nil),
			},
			params: []*corev1.ConfigMap{
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
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-native-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-param-binding", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
						Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-param-binding", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"multi-param-binding-param": "true",
						},
					},
					Namespace: "default",
				}, nil),
			},
			params: []*corev1.ConfigMap{
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
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-binding-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				withMutatingVariables([]v1alpha1.Variable{
					{Name: "foo1", Expression: `"foo1" + "Value"`},
					{Name: "foo2", Expression: `variables.foo1.replace("1", "2")`},
				},
					mutatingPolicy("policy-with-multiple-mutations", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil,
						v1alpha1.Mutation{
							PatchType: v1alpha1.PatchTypeApplyConfiguration,
							ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
						v1alpha1.Mutation{
							PatchType: v1alpha1.PatchTypeJSONPatch,
							JSONPatch: &v1alpha1.JSONPatch{
								Expression: `[
									JSONPatch{op: "test", path: "/metadata/annotations", value: {"foo1": variables.foo1}},
									JSONPatch{op: "add", path: "/metadata/annotations/foo2", value: variables.foo2},
								]`,
							},
						},
						v1alpha1.Mutation{
							PatchType: v1alpha1.PatchTypeApplyConfiguration,
							ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-with-multiple-mutations", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "policy-with-multiple-mutations-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]v1alpha1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-match-condition", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-match-condition", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test"},
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("policy-1", v1alpha1.IfNeededReinvocationPolicy, matchEndpointResources, nil,
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
				mutatingPolicy("policy-2", v1alpha1.IfNeededReinvocationPolicy, matchEndpointResources, nil,
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
				mutatingPolicy("policy-3", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil,
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-1", nil, nil),
				mutatingBinding("policy-2", nil, nil),
				mutatingBinding("policy-3", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &corev1.Endpoints{
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
	flags := []string{fmt.Sprintf("--runtime-config=%s=true", v1alpha1.SchemeGroupVersion)}
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
				_, err = client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Create(ctx, p, metav1.CreateOptions{FieldManager: "integration-test"})
				require.NoError(t, err)
			}

			for _, b := range tc.bindings {
				// After creating each binding, wait until a marker request is successfully mutated.
				err = createAndWaitReadyMutating(ctx, t, client, b, fmt.Sprintf("%d-%s", i, b.Spec.PolicyName))
				require.NoError(t, err)
			}

			unstructuredRequestObj := toUnstructured(t, tc.requestObject)
			unstructuredExpectedObj := toUnstructured(t, tc.expected)
			wipeUncheckedFields(t, unstructuredExpectedObj)

			defer func() {
				if cleanupErr := cleanupMutatingPolicy(ctx, t, client, tc.policies, tc.bindings, tc.params); cleanupErr != nil {
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
}

// TestMutatingAdmissionPolicy_Slow tests policies by waiting until a request is successfully mutated.
// This is slower because it creates an apiserver for each whereas TestMutatingAdmissionPolicy creates
// a single apiserver and then uses marker requests to check that a binding is ready before testing it exactly once.
// Only test cases that cannot be run in TestMutatingAdmissionPolicy should be added here.
func TestMutatingAdmissionPolicy_Slow(t *testing.T) {
	matchEndpointResources := v1alpha1.MatchResources{
		ResourceRules: []v1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: v1alpha1.RuleWithOperations{
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
		policies []*v1alpha1.MutatingAdmissionPolicy
		bindings []*v1alpha1.MutatingAdmissionPolicyBinding
		params   []*corev1.ConfigMap

		requestOperation admissionregistrationv1.OperationType
		requestResource  schema.GroupVersionResource
		subresources     []string       // Only supported for requestOperation=Update since subresources can not be created
		initialObject    runtime.Object // For requestOperation=Update, this may be used to create the initial object state
		requestObject    runtime.Object
		expected         runtime.Object
	}{
		{
			name: "unbound policy is no-op",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("unbound-policy", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
		},
		{
			name: "failure policy ignore",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				withMutatingFailurePolicy(v1alpha1.Ignore,
					mutatingPolicy("policy", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{spec: Object.spec{invalidField: "invalid apply configuration"}}`,
						},
					})),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
		},
		{
			name: "match condition does not match",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]v1alpha1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-no-match-condition", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-no-match-condition", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "production"},
				},
			},
			expected: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "production"},
				},
			},
		},
		{
			name: "some policy conditions match",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				withMutatingMatchConditions([]v1alpha1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("production")`}},
					mutatingPolicy("policy-1", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "wrong"}}}`,
						},
					})),
				withMutatingMatchConditions([]v1alpha1.MatchCondition{{Name: "test-only", Expression: `object.metadata.?labels["environment"] == optional.of("test")`}},
					mutatingPolicy("policy-2", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, nil, v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{labels: {"applied": "updated"}}}`,
						},
					})),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("policy-1", nil, nil),
				mutatingBinding("policy-2", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test"},
				},
			},
			expected: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Labels:    map[string]string{"environment": "test", "applied": "updated"},
				},
			},
		},
		{
			name: "mutate status subresource",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("subresource-status", v1alpha1.NeverReinvocationPolicy, v1alpha1.MatchResources{
					ResourceRules: []v1alpha1.NamedRuleWithOperations{
						{
							RuleWithOperations: v1alpha1.RuleWithOperations{
								Operations: []admissionregistrationv1.OperationType{"*"},
								Rule: admissionregistrationv1.Rule{
									APIGroups:   []string{""},
									APIVersions: []string{"v1"},
									Resources:   []string{"namespaces/status"},
								},
							},
						},
					},
				}, nil, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("subresource-status", nil, nil),
			},
			requestOperation: admissionregistrationv1.Update,
			subresources:     []string{"status"},
			requestResource:  corev1.SchemeGroupVersion.WithResource("namespaces"),
			initialObject: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-namespace",
				},
			},
			requestObject: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-namespace",
				},
				Status: corev1.NamespaceStatus{
					Conditions: []corev1.NamespaceCondition{{
						Type:   corev1.NamespaceDeletionContentFailure,
						Status: corev1.ConditionUnknown,
					}},
				},
			},
			expected: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test-namespace",
					Labels: map[string]string{"kubernetes.io/metadata.name": "test-namespace"},
				},
				Spec: corev1.NamespaceSpec{
					Finalizers: []corev1.FinalizerName{"kubernetes"},
				},
				Status: corev1.NamespaceStatus{
					Conditions: []corev1.NamespaceCondition{{
						Type:    corev1.NamespaceDeletionContentFailure,
						Status:  corev1.ConditionUnknown,
						Message: "mutated",
					}},
					Phase: corev1.NamespaceActive,
				},
			},
		},
		{
			name: "multiple bindings with different params",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy("multi-binding", v1alpha1.NeverReinvocationPolicy, matchEndpointResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType: v1alpha1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &v1alpha1.ApplyConfiguration{
						Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-binding", &v1alpha1.ParamRef{
					Name:      "multi-binding-param-1",
					Namespace: "default",
				}, nil),
				mutatingBinding("multi-binding", &v1alpha1.ParamRef{
					Name:      "multi-binding-param-2",
					Namespace: "default",
				}, nil),
			},
			params: []*corev1.ConfigMap{
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
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-binding-object",
					Namespace: "default",
				},
			},
			expected: &corev1.Endpoints{
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
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				mutatingPolicy(
					"multi-param-reinvocation",
					v1alpha1.IfNeededReinvocationPolicy,
					matchEndpointResources,
					&v1alpha1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
						},
					},
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
					v1alpha1.NeverReinvocationPolicy,
					matchEndpointResources,
					&v1alpha1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
							Expression: `Object{metadata: Object.metadata{annotations: params.data}}`,
						},
					},
					v1alpha1.Mutation{
						PatchType: v1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &v1alpha1.ApplyConfiguration{
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
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				mutatingBinding("multi-param-reinvocation", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"multi-param-reinvocation": "true",
						},
					},
				}, nil),
				mutatingBinding("policy-with-param-no-reinvoke", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Name: "multi-param-reinvocation-param-3",
				}, nil),
			},
			params: []*corev1.ConfigMap{
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
			requestResource:  corev1.SchemeGroupVersion.WithResource("endpoints"),
			requestObject: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-param-reinvocation-object",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &corev1.Endpoints{
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

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
			flags := []string{fmt.Sprintf("--runtime-config=%s=true", v1alpha1.SchemeGroupVersion)}
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
				_, err = client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Create(ctx, p, metav1.CreateOptions{FieldManager: "integration-test"})
				require.NoError(t, err)
			}

			for _, b := range tc.bindings {
				_, err = client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings().Create(ctx, b, metav1.CreateOptions{FieldManager: "integration-test"})
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

// Test_MutatingAdmissionPolicy_CustomResources tests a custom resource mutation.
// CRDs are also ideal for testing version conversion since old version are not removed, so version conversion is also
// tested.
func Test_MutatingAdmissionPolicy_CustomResources(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
	flags := []string{fmt.Sprintf("--runtime-config=%s=true", v1alpha1.SchemeGroupVersion)}
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

	policy := withMutatingFailurePolicy(v1alpha1.Fail, mutatingPolicy("match-by-match-policy-equivalent", v1alpha1.IfNeededReinvocationPolicy, v1alpha1.MatchResources{
		ResourceRules: []v1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: v1alpha1.RuleWithOperations{
					Operations: []v1alpha1.OperationType{
						"*",
					},
					Rule: v1alpha1.Rule{
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
		v1alpha1.Mutation{
			PatchType: v1alpha1.PatchTypeApplyConfiguration,
			ApplyConfiguration: &v1alpha1.ApplyConfiguration{
				Expression: `Object{ metadata: Object.metadata{ labels: {"mutated-panda": "true"} } }`,
			},
		},
	))
	testID := "policy-equivalent"
	policy = withMutatingWaitReadyConstraintAndExpression(policy, testID)
	if _, err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	policyBinding := mutatingBinding("match-by-match-policy-equivalent", nil, nil)
	if err := createAndWaitReadyMutating(ctx, t, client, policyBinding, testID); err != nil {
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

}

func mutatingPolicy(name string, reinvocationPolicy v1alpha1.ReinvocationPolicyType, matchResources v1alpha1.MatchResources, paramKind *v1alpha1.ParamKind, mutations ...v1alpha1.Mutation) *v1alpha1.MutatingAdmissionPolicy {
	return &v1alpha1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1alpha1.MutatingAdmissionPolicySpec{
			MatchConstraints:   &matchResources,
			ParamKind:          paramKind,
			Mutations:          mutations,
			ReinvocationPolicy: reinvocationPolicy,
		},
	}
}

func withMutatingVariables(variables []v1alpha1.Variable, policy *v1alpha1.MutatingAdmissionPolicy) *v1alpha1.MutatingAdmissionPolicy {
	policy.Spec.Variables = variables
	return policy
}

func withMutatingFailurePolicy(failure v1alpha1.FailurePolicyType, policy *v1alpha1.MutatingAdmissionPolicy) *v1alpha1.MutatingAdmissionPolicy {
	policy.Spec.FailurePolicy = &failure
	return policy
}

func withMutatingMatchConditions(matchConditions []v1alpha1.MatchCondition, policy *v1alpha1.MutatingAdmissionPolicy) *v1alpha1.MutatingAdmissionPolicy {
	policy.Spec.MatchConditions = matchConditions
	return policy
}

func withMutatingWaitReadyConstraintAndExpression(policy *v1alpha1.MutatingAdmissionPolicy, testID string) *v1alpha1.MutatingAdmissionPolicy {
	policy = policy.DeepCopy()
	policy.Spec.MatchConstraints.ResourceRules = append(policy.Spec.MatchConstraints.ResourceRules, v1alpha1.NamedRuleWithOperations{
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
	policy.Spec.Mutations = append([]v1alpha1.Mutation{{
		PatchType: v1alpha1.PatchTypeApplyConfiguration,
		ApplyConfiguration: &v1alpha1.ApplyConfiguration{
			// Only mutate mutation-markers.
			Expression: fmt.Sprintf(`object.metadata.?labels["mutation-marker"] == optional.of("%v") ? Object{ metadata: Object.metadata{ labels: {"mutated":"%v"}}}: Object{}`, testID, testID),
		},
	}}, policy.Spec.Mutations...)
	return policy
}

func mutatingBinding(policyName string, paramRef *v1alpha1.ParamRef, matchResources *v1alpha1.MatchResources) *v1alpha1.MutatingAdmissionPolicyBinding {
	return &v1alpha1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: policyName + "-binding-" + string(uuid.NewUUID()),
		},
		Spec: v1alpha1.MutatingAdmissionPolicyBindingSpec{
			PolicyName:     policyName,
			ParamRef:       paramRef,
			MatchResources: matchResources,
		},
	}
}

func createAndWaitReadyMutating(ctx context.Context, t *testing.T, client kubernetes.Interface, binding *v1alpha1.MutatingAdmissionPolicyBinding, testID string) error {
	return createAndWaitReadyNamespacedWithWarnHandlerMutating(ctx, t, client, binding, "default", testID)
}

func createAndWaitReadyNamespacedWithWarnHandlerMutating(ctx context.Context, t *testing.T, client kubernetes.Interface, binding *v1alpha1.MutatingAdmissionPolicyBinding, ns string, testID string) error {
	_, err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	marker := &corev1.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "test-marker", Namespace: ns, Labels: map[string]string{"mutation-marker": testID}}}
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

func cleanupMutatingPolicy(ctx context.Context, t *testing.T, client kubernetes.Interface, policies []*v1alpha1.MutatingAdmissionPolicy, bindings []*v1alpha1.MutatingAdmissionPolicyBinding, params []*corev1.ConfigMap) error {
	for _, policy := range policies {
		if err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Delete(ctx, policy.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
			_, err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Get(ctx, policy.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}); waitErr != nil {
			t.Fatalf("timed out waiting for policy to be cleaned up: %v", waitErr)
		}
	}

	for _, binding := range bindings {
		err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings().Delete(ctx, binding.Name, metav1.DeleteOptions{})
		if err != nil {
			t.Fatal(err)
		}

		if waitErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
			_, err := client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings().Get(ctx, binding.Name, metav1.GetOptions{})
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
			if err := client.CoreV1().ConfigMaps(param.GetNamespace()).Delete(ctx, param.Name, metav1.DeleteOptions{}); err != nil {
				if apierrors.IsNotFound(err) {
					return true, nil
				}
				return false, nil
			}
			_, err := client.CoreV1().ConfigMaps(param.GetNamespace()).Get(ctx, param.Name, metav1.GetOptions{})
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
