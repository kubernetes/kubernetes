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
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/restmapper"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	coreinstall "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func policy(name string, matchResources v1alpha1.MatchResources, paramKind *v1alpha1.ParamKind, mutations ...v1alpha1.Mutation) *v1alpha1.MutatingAdmissionPolicy {
	return &v1alpha1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1alpha1.MutatingAdmissionPolicySpec{
			MatchConstraints: &matchResources,
			ParamKind:        paramKind,
			Mutations:        mutations,
		},
	}
}

func binding(policyName string, paramRef *v1alpha1.ParamRef, matchResources *v1alpha1.MatchResources) *v1alpha1.MutatingAdmissionPolicyBinding {
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

func TestMAP(t *testing.T) {
	matchAllResources := v1alpha1.MatchResources{
		ResourceRules: []v1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: v1alpha1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{"*"},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"*"},
						APIVersions: []string{"*"},
						Resources:   []string{"*"},
					},
				},
			},
		},
	}

	cases := []struct {
		name     string
		policies []*v1alpha1.MutatingAdmissionPolicy
		bindings []*v1alpha1.MutatingAdmissionPolicyBinding
		params   []runtime.Object

		requestOperation admissionregistrationv1.OperationType
		requestResource  schema.GroupVersionResource
		requestObject    runtime.Object

		expected runtime.Object
	}{
		{
			name: "basic",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("basic-policy", matchAllResources, nil, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"my-foo-annotation": "myAnnotationValue"
								}
							}
						}`,
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("basic-policy", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
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
				policy("policy-1", matchAllResources, nil, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo1": "foo1Value"
								}
							}
						}`,
				}),
				policy("policy-2", matchAllResources, nil, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo2": "foo2Value"
								}
							}
						}`,
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-1", nil, nil),
				binding("policy-2", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
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
				policy("policy-with-param", matchAllResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									params.data["key"]: params.data["value"]
								}
							}
						}`,
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-param", &v1alpha1.ParamRef{
					Name: "test-param",
				}, nil),
			},
			params: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param",
						Namespace: "default",
					},
					Data: map[string]string{
						"key":   "myFooKey",
						"value": "myFooValue",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"myFooKey": "myFooValue",
					},
				},
			},
		},
		// {
		// Can't be reliably tested due to large possibly 30s delay in picking up
		// CRDs in discovery
		// 	name: "policy with crd param",
		// },
		{
			name: "multiple bindings with different params",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("policy-with-param", matchAllResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression:         `Object{metadata: Object.metadata{annotations: params.data}}`,
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-param", &v1alpha1.ParamRef{
					Name:      "test-param-1",
					Namespace: "default",
				}, nil),
				binding("policy-with-param", &v1alpha1.ParamRef{
					Name:      "test-param-2",
					Namespace: "default",
				}, nil),
			},
			params: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-1",
						Namespace: "default",
					},
					Data: map[string]string{
						"key1": "value1",
					},
				},
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-2",
						Namespace: "default",
					},
					Data: map[string]string{
						"key2": "value2",
						"key3": "value3",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"key1": "value1",
						"key2": "value2",
						"key3": "value3",
					},
				},
			},
		},
		{
			name: "policy with multiple params quantified by single binding",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("policy-with-param", matchAllResources, &v1alpha1.ParamKind{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				}, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression:         `Object{metadata: Object.metadata{annotations: params.data}}`,
				}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-param", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"param": "true",
						},
					},
				}, nil),
			},
			params: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-1",
						Namespace: "default",
						Labels: map[string]string{
							"param": "true",
						},
					},
					Data: map[string]string{
						"key1": "value1",
					},
				},
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-2",
						Namespace: "default",
						Labels: map[string]string{
							"param": "true",
						},
					},
					Data: map[string]string{
						"key2": "value2",
						"key3": "value3",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"key1": "value1",
						"key2": "value2",
						"key3": "value3",
					},
				},
			},
		},
		{
			name: "policy with multiple mutations",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("policy-with-multiple-mutations", matchAllResources, nil,
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"foo1": "foo1Value"
									}
								}
							}`,
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"foo2": "foo2Value"
									}
								}
							}`,
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"foo3": "foo3Value"
									}
								}
							}`,
					},
				),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-multiple-mutations", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
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
			name: "policy with multiple mutations requiring reinvocation",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("policy-with-multiple-mutations-requiring-reinvocation", matchAllResources, nil,
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
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
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
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
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
										"thirdApplied": "true"
									}
								}
							}`,
					}),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-multiple-mutations-requiring-reinvocation", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &corev1.ConfigMap{
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
		{
			// same as the multiple mutations test, but the mutations are split
			// across multiple policies
			name: "multiple policies requiring reinvocation",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("policy-1", matchAllResources, nil,
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
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
				),
				policy("policy-2", matchAllResources, nil,
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
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
				),
				policy("policy-3", matchAllResources, nil,
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
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
				),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-1", nil, nil),
				binding("policy-2", nil, nil),
				binding("policy-3", nil, nil),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &corev1.ConfigMap{
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
		{
			// Same as the other cases, but the reinvocation is caused by
			// multiple params bound
			name: "multiple params causing reinvocation",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy(
					"policy-with-param",
					matchAllResources,
					&v1alpha1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
						Expression:         `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.IfNeededReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
									}
								}
							}`,
					},
				),
				policy(
					"policy-with-param-no-reinvoke",
					matchAllResources,
					&v1alpha1.ParamKind{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression:         `Object{metadata: Object.metadata{annotations: params.data}}`,
					},
					v1alpha1.Mutation{
						PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
						ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										?"foo": optional.of(string(int(object.metadata.annotations["foo"]) + 1)),
									}
								}
							}`,
					},
				),
			},
			bindings: []*v1alpha1.MutatingAdmissionPolicyBinding{
				binding("policy-with-param", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"param": "true",
						},
					},
				}, nil),
				binding("policy-with-param-no-reinvoke", &v1alpha1.ParamRef{
					// note empty namespace. all params matching request namespace
					// will be used
					Name: "test-param-3",
				}, nil),
			},
			params: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-1",
						Namespace: "default",
						Labels: map[string]string{
							"param": "true",
						},
					},
					Data: map[string]string{
						"firstApplied": "true",
					},
				},
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-2",
						Namespace: "default",
						Labels: map[string]string{
							"param": "true",
						},
					},
					Data: map[string]string{
						"secondApplied": "true",
					},
				},
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-param-3",
						Namespace: "default",
					},
					Data: map[string]string{
						"thirdApplied": "true",
					},
				},
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
					Namespace: "default",
					Annotations: map[string]string{
						"foo": "0",
					},
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "request-configmap",
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
		{
			name: "unbound policy is no-op",
			policies: []*v1alpha1.MutatingAdmissionPolicy{
				policy("unbound-policy", matchAllResources, nil, v1alpha1.Mutation{
					PatchType:          ptr.To(v1alpha1.ApplyConfigurationPatchType),
					ReinvocationPolicy: ptr.To(v1alpha1.NeverReinvocationPolicy),
					Expression: `
						Object{
							metadata: Object.metadata{
								annotations: {
									"foo": "fooValue"
								}
							}
						}`,
				}),
			},
			requestOperation: admissionregistrationv1.Create,
			requestResource:  corev1.SchemeGroupVersion.WithResource("configmaps"),
			requestObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-configmap",
					Namespace: "default",
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.requestObject == nil {
				t.Fatalf("requestObject must be set")
			} else if tc.expected == nil {
				t.Fatalf("expected must be set")
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.MutatingAdmissionPolicy, true)
			server, err := apiservertesting.StartTestServer(t, nil, []string{
				"--enable-admission-plugins", "MutatingAdmissionPolicy",
			}, framework.SharedEtcd())
			require.NoError(t, err)
			defer server.TearDownFn()

			client, err := clientset.NewForConfig(server.ClientConfig)
			require.NoError(t, err)

			dynClient, err := dynamic.NewForConfig(server.ClientConfig)
			require.NoError(t, err)

			restMapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(client.DiscoveryClient))

			sch := runtime.NewScheme()
			coreinstall.Install(sch)

			guessGVK := func(obj runtime.Object) schema.GroupVersionKind {
				gvk := obj.GetObjectKind().GroupVersionKind()
				if gvk.Empty() {
					// Try to read gvk off of the schema
					gvks, _, err := sch.ObjectKinds(obj)
					require.NoError(t, err)
					require.NotEmpty(t, gvks)

					gvk = gvks[0]
				}
				return gvk
			}

			for _, p := range tc.policies {
				_, err = client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies().Create(context.TODO(), p, metav1.CreateOptions{FieldManager: "integration-test"})
				require.NoError(t, err)
			}

			for _, b := range tc.bindings {
				_, err = client.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings().Create(context.TODO(), b, metav1.CreateOptions{FieldManager: "integration-test"})
				require.NoError(t, err)
			}

			for _, obj := range tc.params {
				gvk := guessGVK(obj)

				mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
				require.NoError(t, err)

				unstructuredParamMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
				require.NoError(t, err)

				unstructuredParamObj := &unstructured.Unstructured{
					Object: unstructuredParamMap,
				}

				var rsrcClient dynamic.ResourceInterface = dynClient.Resource(mapping.Resource)
				if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
					rsrcClient = rsrcClient.(dynamic.NamespaceableResourceInterface).Namespace(unstructuredParamObj.GetNamespace())
				}
				_, err = rsrcClient.Create(context.TODO(), unstructuredParamObj, metav1.CreateOptions{FieldManager: "integration-test"})
				require.NoError(t, err)
			}

			// Convert request object to unstructured
			unstructuredRequestMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(tc.requestObject)
			require.NoError(t, err)

			unstructuredRequestObj := &unstructured.Unstructured{
				Object: unstructuredRequestMap,
			}

			unstructuredExpectedMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(tc.expected)
			require.NoError(t, err)

			unstructuredExpectedObj := &unstructured.Unstructured{
				Object: unstructuredExpectedMap,
			}

			wipeUncheckedFields := func(obj runtime.Object) {
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
			wipeUncheckedFields(unstructuredExpectedObj)

			// Dry Run the request until we get the expected/mutated response
			var resultObj runtime.Object
			err = wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (done bool, err error) {
				acc, err := meta.Accessor(tc.requestObject)
				require.NoError(t, err)

				var rsrcClient dynamic.ResourceInterface = dynClient.Resource(tc.requestResource)
				if len(acc.GetNamespace()) > 0 {
					rsrcClient = rsrcClient.(dynamic.NamespaceableResourceInterface).Namespace(acc.GetNamespace())
				}

				switch tc.requestOperation {
				case admissionregistrationv1.Create:
					resultObj, err = rsrcClient.Create(context.TODO(), unstructuredRequestObj, metav1.CreateOptions{
						DryRun:       []string{metav1.DryRunAll},
						FieldManager: "integration-test",
					})
					if err != nil {
						return false, err
					}
					wipeUncheckedFields(resultObj)
					return reflect.DeepEqual(unstructuredExpectedObj, resultObj), nil
				case admissionregistrationv1.Update:
					resultObj, err = rsrcClient.Update(context.TODO(), unstructuredRequestObj, metav1.UpdateOptions{
						DryRun:       []string{metav1.DryRunAll},
						FieldManager: "integration-test",
					})
					if err != nil {
						return false, err
					}
					wipeUncheckedFields(resultObj)
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
