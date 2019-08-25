// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resmaptest"
	"sigs.k8s.io/kustomize/pkg/resource"
)

func TestNameReferenceHappyRun(t *testing.T) {
	rf := resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl())
	m := resmaptest_test.NewRmBuilder(t, rf).AddWithName(
		"cm1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "someprefix-cm1-somehash",
			},
		}).AddWithName(
		"cm2",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "someprefix-cm2-somehash",
			},
		}).AddWithName(
		"secret1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Secret",
			"metadata": map[string]interface{}{
				"name": "someprefix-secret1-somehash",
			},
		}).AddWithName(
		"claim1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PersistentVolumeClaim",
			"metadata": map[string]interface{}{
				"name": "someprefix-claim1",
			},
		}).Add(
		map[string]interface{}{
			"group":      "extensions",
			"apiVersion": "v1beta1",
			"kind":       "Ingress",
			"metadata": map[string]interface{}{
				"name": "ingress1",
				"annotations": map[string]interface{}{
					"ingress.kubernetes.io/auth-secret":       "secret1",
					"nginx.ingress.kubernetes.io/auth-secret": "secret1",
				},
			},
			"spec": map[string]interface{}{
				"backend": map[string]interface{}{
					"serviceName": "testsvc",
					"servicePort": "80",
				},
			},
		},
	).Add(
		map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
								"env": []interface{}{
									map[string]interface{}{
										"name": "CM_FOO",
										"valueFrom": map[string]interface{}{
											"configMapKeyRef": map[string]interface{}{
												"name": "cm1",
												"key":  "somekey",
											},
										},
									},
									map[string]interface{}{
										"name": "SECRET_FOO",
										"valueFrom": map[string]interface{}{
											"secretKeyRef": map[string]interface{}{
												"name": "secret1",
												"key":  "somekey",
											},
										},
									},
								},
								"envFrom": []interface{}{
									map[string]interface{}{
										"configMapRef": map[string]interface{}{
											"name": "cm1",
											"key":  "somekey",
										},
									},
									map[string]interface{}{
										"secretRef": map[string]interface{}{
											"name": "secret1",
											"key":  "somekey",
										},
									},
								},
							},
						},
						"imagePullSecrets": []interface{}{
							map[string]interface{}{
								"name": "secret1",
							},
						},
						"volumes": map[string]interface{}{
							"configMap": map[string]interface{}{
								"name": "cm1",
							},
							"projected": map[string]interface{}{
								"sources": map[string]interface{}{
									"configMap": map[string]interface{}{
										"name": "cm2",
									},
									"secret": map[string]interface{}{
										"name": "secret1",
									},
								},
							},
							"secret": map[string]interface{}{
								"secretName": "secret1",
							},
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": "claim1",
							},
						},
					},
				},
			},
		}).Add(
		map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "StatefulSet",
			"metadata": map[string]interface{}{
				"name": "statefulset1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
						"volumes": map[string]interface{}{
							"projected": map[string]interface{}{
								"sources": map[string]interface{}{
									"configMap": map[string]interface{}{
										"name": "cm2",
									},
									"secret": map[string]interface{}{
										"name": "secret1",
									},
								},
							},
						},
					},
				},
			},
		}).AddWithName("sa",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ServiceAccount",
			"metadata": map[string]interface{}{
				"name":      "someprefix-sa",
				"namespace": "test",
			},
		}).Add(
		map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRoleBinding",
			"metadata": map[string]interface{}{
				"name": "crb",
			},
			"subjects": []interface{}{
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "sa",
					"namespace": "test",
				},
			},
		}).Add(
		map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRole",
			"metadata": map[string]interface{}{
				"name": "cr",
			},
			"rules": []interface{}{
				map[string]interface{}{
					"resources": []interface{}{
						"secrets",
					},
					"resourceNames": []interface{}{
						"secret1",
						"secret1",
						"secret2",
					},
				},
			},
		}).Add(
		map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob1",
			},
			"spec": map[string]interface{}{
				"schedule": "0 14 * * *",
				"jobTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"template": map[string]interface{}{
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "main",
										"image": "myimage",
									},
								},
								"volumes": map[string]interface{}{
									"projected": map[string]interface{}{
										"sources": map[string]interface{}{
											"configMap": map[string]interface{}{
												"name": "cm2",
											},
											"secret": map[string]interface{}{
												"name": "secret1",
											},
										},
									},
								},
							},
						},
					},
				},
			},
		}).ResMap()

	expected := resmaptest_test.NewSeededRmBuilder(t, rf, m.ShallowCopy()).ReplaceResource(
		map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
								"env": []interface{}{
									map[string]interface{}{
										"name": "CM_FOO",
										"valueFrom": map[string]interface{}{
											"configMapKeyRef": map[string]interface{}{
												"name": "someprefix-cm1-somehash",
												"key":  "somekey",
											},
										},
									},
									map[string]interface{}{
										"name": "SECRET_FOO",
										"valueFrom": map[string]interface{}{
											"secretKeyRef": map[string]interface{}{
												"name": "someprefix-secret1-somehash",
												"key":  "somekey",
											},
										},
									},
								},
								"envFrom": []interface{}{
									map[string]interface{}{
										"configMapRef": map[string]interface{}{
											"name": "someprefix-cm1-somehash",
											"key":  "somekey",
										},
									},
									map[string]interface{}{
										"secretRef": map[string]interface{}{
											"name": "someprefix-secret1-somehash",
											"key":  "somekey",
										},
									},
								},
							},
						},
						"imagePullSecrets": []interface{}{
							map[string]interface{}{
								"name": "someprefix-secret1-somehash",
							},
						},
						"volumes": map[string]interface{}{
							"configMap": map[string]interface{}{
								"name": "someprefix-cm1-somehash",
							},
							"projected": map[string]interface{}{
								"sources": map[string]interface{}{
									"configMap": map[string]interface{}{
										"name": "someprefix-cm2-somehash",
									},
									"secret": map[string]interface{}{
										"name": "someprefix-secret1-somehash",
									},
								},
							},
							"secret": map[string]interface{}{
								"secretName": "someprefix-secret1-somehash",
							},
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": "someprefix-claim1",
							},
						},
					},
				},
			},
		}).ReplaceResource(
		map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "StatefulSet",
			"metadata": map[string]interface{}{
				"name": "statefulset1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
						"volumes": map[string]interface{}{
							"projected": map[string]interface{}{
								"sources": map[string]interface{}{
									"configMap": map[string]interface{}{
										"name": "someprefix-cm2-somehash",
									},
									"secret": map[string]interface{}{
										"name": "someprefix-secret1-somehash",
									},
								},
							},
						},
					},
				},
			},
		}).ReplaceResource(
		map[string]interface{}{
			"group":      "extensions",
			"apiVersion": "v1beta1",
			"kind":       "Ingress",
			"metadata": map[string]interface{}{
				"name": "ingress1",
				"annotations": map[string]interface{}{
					"ingress.kubernetes.io/auth-secret":       "someprefix-secret1-somehash",
					"nginx.ingress.kubernetes.io/auth-secret": "someprefix-secret1-somehash",
				},
			},
			"spec": map[string]interface{}{
				"backend": map[string]interface{}{
					"serviceName": "testsvc",
					"servicePort": "80",
				},
			},
		}).ReplaceResource(
		map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRoleBinding",
			"metadata": map[string]interface{}{
				"name": "crb",
			},
			"subjects": []interface{}{
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "someprefix-sa",
					"namespace": "test",
				},
			},
		}).ReplaceResource(
		map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRole",
			"metadata": map[string]interface{}{
				"name": "cr",
			},
			"rules": []interface{}{
				map[string]interface{}{
					"resources": []interface{}{
						"secrets",
					},
					"resourceNames": []interface{}{
						"someprefix-secret1-somehash",
						"someprefix-secret1-somehash",
						"secret2",
					},
				},
			},
		}).ReplaceResource(
		map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob1",
			},
			"spec": map[string]interface{}{
				"schedule": "0 14 * * *",
				"jobTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"template": map[string]interface{}{
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "main",
										"image": "myimage",
									},
								},
								"volumes": map[string]interface{}{
									"projected": map[string]interface{}{
										"sources": map[string]interface{}{
											"configMap": map[string]interface{}{
												"name": "someprefix-cm2-somehash",
											},
											"secret": map[string]interface{}{
												"name": "someprefix-secret1-somehash",
											},
										},
									},
								},
							},
						},
					},
				},
			},
		}).ResMap()

	nrt := NewNameReferenceTransformer(defaultTransformerConfig.NameReference)
	err := nrt.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestNameReferenceUnhappyRun(t *testing.T) {
	rf := resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl())
	tests := []struct {
		resMap      resmap.ResMap
		expectedErr string
	}{
		{
			resMap: resmaptest_test.NewRmBuilder(t, rf).Add(
				map[string]interface{}{
					"apiVersion": "rbac.authorization.k8s.io/v1",
					"kind":       "ClusterRole",
					"metadata": map[string]interface{}{
						"name": "cr",
					},
					"rules": []interface{}{
						map[string]interface{}{
							"resources": []interface{}{
								"secrets",
							},
							"resourceNames": []interface{}{
								[]interface{}{},
							},
						},
					},
				}).ResMap(),
			expectedErr: "is expected to be string"},
		{
			resMap: resmaptest_test.NewRmBuilder(t, rf).Add(
				map[string]interface{}{
					"apiVersion": "rbac.authorization.k8s.io/v1",
					"kind":       "ClusterRole",
					"metadata": map[string]interface{}{
						"name": "cr",
					},
					"rules": []interface{}{
						map[string]interface{}{
							"resources": []interface{}{
								"secrets",
							},
							"resourceNames": map[string]interface{}{
								"foo": "bar",
							},
						},
					},
				}).ResMap(),
			expectedErr: "is expected to be either a string or a []interface{}"},
	}

	nrt := NewNameReferenceTransformer(defaultTransformerConfig.NameReference)
	for _, test := range tests {
		err := nrt.Transform(test.resMap)
		if err == nil {
			t.Fatalf("expected error to happen")
		}

		if !strings.Contains(err.Error(), test.expectedErr) {
			t.Fatalf("Incorrect error.\nExpected: %s, but got %v",
				test.expectedErr, err)
		}
	}
}

func TestNameReferencePersistentVolumeHappyRun(t *testing.T) {
	rf := resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl())

	v1 := rf.FromMapWithName(
		"volume1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PersistentVolume",
			"metadata": map[string]interface{}{
				"name": "someprefix-volume1",
			},
		})
	c1 := rf.FromMapWithName(
		"claim1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PersistentVolumeClaim",
			"metadata": map[string]interface{}{
				"name":      "someprefix-claim1",
				"namespace": "some-namespace",
			},
			"spec": map[string]interface{}{
				"volumeName": "volume1",
			},
		})

	v2 := v1.DeepCopy()
	c2 := rf.FromMapWithName(
		"claim1",
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PersistentVolumeClaim",
			"metadata": map[string]interface{}{
				"name":      "someprefix-claim1",
				"namespace": "some-namespace",
			},
			"spec": map[string]interface{}{
				"volumeName": "someprefix-volume1",
			},
		})

	m1 := resmaptest_test.NewRmBuilder(t, rf).AddR(v1).AddR(c1).ResMap()

	nrt := NewNameReferenceTransformer(defaultTransformerConfig.NameReference)
	if err := nrt.Transform(m1); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m2 := resmaptest_test.NewRmBuilder(t, rf).AddR(v2).AddR(c2).ResMap()
	v2.AppendRefBy(c2.CurId())

	if err := m1.ErrorIfNotEqualLists(m2); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}
