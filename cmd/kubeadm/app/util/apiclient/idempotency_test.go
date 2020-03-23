/*
Copyright 2018 The Kubernetes Authors.

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

package apiclient

import (
	"context"
	"testing"

	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

const configMapName = "configmap"

func TestPatchNodeNonErrorCases(t *testing.T) {
	testcases := []struct {
		name       string
		lookupName string
		node       v1.Node
		success    bool
	}{
		{
			name:       "simple update",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success: true,
		},
		{
			name:       "node does not exist",
			lookupName: "whale",
			success:    false,
		},
		{
			name:       "node not labelled yet",
			lookupName: "robin",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "robin",
				},
			},
			success: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			_, err := client.CoreV1().Nodes().Create(context.TODO(), &tc.node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create node to fake client: %v", err)
			}
			conditionFunction := PatchNodeOnce(client, tc.lookupName, func(node *v1.Node) {
				node.Annotations = map[string]string{
					"updatedBy": "test",
				}
			})
			success, err := conditionFunction()
			if err != nil {
				t.Fatalf("did not expect error: %v", err)
			}
			if success != tc.success {
				t.Fatalf("expected %v got %v", tc.success, success)
			}
		})
	}
}

func TestCreateOrMutateConfigMap(t *testing.T) {
	client := fake.NewSimpleClientset()
	err := CreateOrMutateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"key": "some-value",
		},
	}, func(cm *v1.ConfigMap) error {
		t.Fatal("mutate should not have been called, since the ConfigMap should have been created instead of mutated")
		return nil
	})
	if err != nil {
		t.Fatalf("error creating ConfigMap: %v", err)
	}
	_, err = client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), configMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error retrieving ConfigMap: %v", err)
	}
}

func createClientAndConfigMap(t *testing.T) *fake.Clientset {
	client := fake.NewSimpleClientset()
	_, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"key": "some-value",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating ConfigMap: %v", err)
	}
	return client
}

func TestMutateConfigMap(t *testing.T) {
	client := createClientAndConfigMap(t)

	err := MutateConfigMap(client, metav1.ObjectMeta{
		Name:      configMapName,
		Namespace: metav1.NamespaceSystem,
	}, func(cm *v1.ConfigMap) error {
		cm.Data["key"] = "some-other-value"
		return nil
	})
	if err != nil {
		t.Fatalf("error mutating regular ConfigMap: %v", err)
	}

	cm, _ := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), configMapName, metav1.GetOptions{})
	if cm.Data["key"] != "some-other-value" {
		t.Fatalf("ConfigMap mutation was invalid, has: %q", cm.Data["key"])
	}
}

func TestMutateConfigMapWithConflict(t *testing.T) {
	client := createClientAndConfigMap(t)

	// Mimic that the first 5 updates of the ConfigMap returns a conflict, whereas the sixth update
	// succeeds
	conflict := 5
	client.PrependReactor("update", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		if conflict > 0 {
			conflict--
			return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), configMapName, errors.New("conflict"))
		}
		return false, update.GetObject(), nil
	})

	err := MutateConfigMap(client, metav1.ObjectMeta{
		Name:      configMapName,
		Namespace: metav1.NamespaceSystem,
	}, func(cm *v1.ConfigMap) error {
		cm.Data["key"] = "some-other-value"
		return nil
	})
	if err != nil {
		t.Fatalf("error mutating conflicting ConfigMap: %v", err)
	}

	cm, _ := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), configMapName, metav1.GetOptions{})
	if cm.Data["key"] != "some-other-value" {
		t.Fatalf("ConfigMap mutation with conflict was invalid, has: %q", cm.Data["key"])
	}
}
