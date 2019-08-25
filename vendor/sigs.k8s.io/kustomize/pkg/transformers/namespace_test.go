// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"testing"

	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/resmaptest"
	"sigs.k8s.io/kustomize/pkg/resource"
)

func TestNamespaceRun(t *testing.T) {
	rf := resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl())
	m := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":      "cm2",
				"namespace": "foo",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Namespace",
			"metadata": map[string]interface{}{
				"name": "ns1",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ServiceAccount",
			"metadata": map[string]interface{}{
				"name":      "default",
				"namespace": "system",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ServiceAccount",
			"metadata": map[string]interface{}{
				"name":      "service-account",
				"namespace": "system",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRoleBinding",
			"metadata": map[string]interface{}{
				"name": "manager-rolebinding",
			},
			"subjects": []interface{}{
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "default",
					"namespace": "system",
				},
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "service-account",
					"namespace": "system",
				},
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "another",
					"namespace": "random",
				},
			}}).
		Add(map[string]interface{}{
			"apiVersion": "apiextensions.k8s.io/v1beta1",
			"kind":       "CustomResourceDefinition",
			"metadata": map[string]interface{}{
				"name": "crd",
			}}).ResMap()

	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":      "cm1",
				"namespace": "test",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":      "cm2",
				"namespace": "test",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Namespace",
			"metadata": map[string]interface{}{
				"name": "ns1",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ServiceAccount",
			"metadata": map[string]interface{}{
				"name":      "default",
				"namespace": "test",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ServiceAccount",
			"metadata": map[string]interface{}{
				"name":      "service-account",
				"namespace": "test",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRoleBinding",
			"metadata": map[string]interface{}{
				"name": "manager-rolebinding",
			},
			"subjects": []interface{}{
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "default",
					"namespace": "test",
				},
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "service-account",
					"namespace": "test",
				},
				map[string]interface{}{
					"kind":      "ServiceAccount",
					"name":      "another",
					"namespace": "random",
				},
			}}).
		Add(map[string]interface{}{
			"apiVersion": "apiextensions.k8s.io/v1beta1",
			"kind":       "CustomResourceDefinition",
			"metadata": map[string]interface{}{
				"name": "crd",
			}}).ResMap()

	nst := NewNamespaceTransformer("test", defaultTransformerConfig.NameSpace)
	err := nst.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestNamespaceRunForClusterLevelKind(t *testing.T) {
	rf := resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl())
	m := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Namespace",
			"metadata": map[string]interface{}{
				"name": "ns1",
			}}).
		Add(map[string]interface{}{
			"kind": "CustomResourceDefinition",
			"metadata": map[string]interface{}{
				"name": "crd1",
			}}).
		Add(map[string]interface{}{
			"kind": "PersistentVolume",
			"metadata": map[string]interface{}{
				"name": "pv1",
			}}).
		Add(map[string]interface{}{
			"kind": "ClusterRole",
			"metadata": map[string]interface{}{
				"name": "cr1",
			}}).
		Add(map[string]interface{}{
			"kind": "ClusterRoleBinding",
			"metadata": map[string]interface{}{
				"name": "crb1",
			},
			"subjects": []interface{}{}}).ResMap()

	expected := m.DeepCopy()

	nst := NewNamespaceTransformer("test", defaultTransformerConfig.NameSpace)

	err := nst.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}
