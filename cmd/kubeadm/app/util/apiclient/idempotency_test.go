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
	"reflect"
	"testing"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

const configMapName = "configmap"

func TestPatchNode(t *testing.T) {
	testcases := []struct {
		name       string
		lookupName string
		node       v1.Node
		success    bool
		fakeError  error
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
		{
			name:       "patch node when timeout",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success:   false,
			fakeError: apierrors.NewTimeoutError("fake timeout", -1),
		},
		{
			name:       "patch node when conflict",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success:   false,
			fakeError: apierrors.NewConflict(schema.GroupResource{}, "fake conflict", nil),
		},
		{
			name:       "patch node when there is a server timeout",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success:   false,
			fakeError: apierrors.NewServerTimeout(schema.GroupResource{}, "fake server timeout", 1),
		},
		{
			name:       "patch node when the service is unavailable",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success:   false,
			fakeError: apierrors.NewServiceUnavailable("fake service unavailable"),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			_, err := client.CoreV1().Nodes().Create(context.TODO(), &tc.node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create node to fake client: %v", err)
			}
			if tc.fakeError != nil {
				client.PrependReactor("patch", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, tc.fakeError
				})
			}
			var lastError error
			conditionFunction := PatchNodeOnce(client, tc.lookupName, func(node *v1.Node) {
				node.Annotations = map[string]string{
					"updatedBy": "test",
				}
			}, &lastError)
			success, err := conditionFunction(context.Background())
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

func TestGetConfigMapWithShortRetry(t *testing.T) {
	type args struct {
		client    clientset.Interface
		namespace string
		name      string
	}
	tests := []struct {
		name    string
		args    args
		want    *v1.ConfigMap
		wantErr bool
	}{
		{
			name: "ConfigMap exists",
			args: args{
				client:    newMockClientForTest(t, "default", "foo", "ConfigMap"),
				namespace: "default",
				name:      "foo",
			},
			want: &v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       configMapName,
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "default",
				},
			},
			wantErr: false,
		},
		{
			name: "ConfigMap does not exist",
			args: args{
				client:    fake.NewSimpleClientset(),
				namespace: "default",
				name:      "foo",
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetConfigMapWithShortRetry(tt.args.client, tt.args.namespace, tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetConfigMapWithShortRetry() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetConfigMapWithShortRetry() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateOrUpdateClusterRole(t *testing.T) {
	testClusterRole := &rbac.ClusterRole{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ClusterRole",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}

	type args struct {
		client      clientset.Interface
		clusterRole *rbac.ClusterRole
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "ClusterRole does not exist",
			args: args{
				client:      fake.NewSimpleClientset(),
				clusterRole: testClusterRole,
			},
			wantErr: false,
		},
		{
			name: "ClusterRole exists",
			args: args{
				client:      newMockClientForTest(t, "", "foo", "ClusterRole"),
				clusterRole: testClusterRole,
			},
			wantErr: false,
		},
		{
			name: "ClusterRole is invalid",
			args: args{
				client:      fake.NewSimpleClientset(),
				clusterRole: nil,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CreateOrUpdateClusterRole(tt.args.client, tt.args.clusterRole); (err != nil) != tt.wantErr {
				t.Errorf("CreateOrUpdateClusterRole() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateOrUpdateClusterRoleBinding(t *testing.T) {
	testClusterRoleBinding := &rbac.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ClusterRoleBinding",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}

	type args struct {
		client             clientset.Interface
		clusterRoleBinding *rbac.ClusterRoleBinding
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "ClusterRoleBinding does not exist",
			args: args{
				client:             fake.NewSimpleClientset(),
				clusterRoleBinding: testClusterRoleBinding,
			},
			wantErr: false,
		},
		{
			name: "ClusterRoleBinding exists",
			args: args{
				client:             newMockClientForTest(t, "", "foo", "ClusterRoleBinding"),
				clusterRoleBinding: testClusterRoleBinding,
			},
			wantErr: false,
		},
		{
			name: "ClusterRoleBinding is invalid",
			args: args{
				client:             fake.NewSimpleClientset(),
				clusterRoleBinding: nil,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CreateOrUpdateClusterRoleBinding(tt.args.client, tt.args.clusterRoleBinding); (err != nil) != tt.wantErr {
				t.Errorf("CreateOrUpdateClusterRoleBinding() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func newMockClientForTest(t *testing.T, namepsace string, name string, kind string) *fake.Clientset {
	client := fake.NewSimpleClientset()

	switch kind {
	case "ConfigMap":
		_, err := client.CoreV1().ConfigMaps(namepsace).Create(context.Background(), &v1.ConfigMap{
			TypeMeta: metav1.TypeMeta{
				Kind:       configMapName,
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: namepsace,
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating ConfigMap: %v", err)
		}
	case "ClusterRole":
		_, err := client.RbacV1().ClusterRoles().Create(context.Background(), &rbac.ClusterRole{
			TypeMeta: metav1.TypeMeta{
				Kind:       "ClusterRole",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating ClusterRole: %v", err)
		}
	case "ClusterRoleBinding":
		_, err := client.RbacV1().ClusterRoleBindings().Create(context.Background(), &rbac.ClusterRoleBinding{
			TypeMeta: metav1.TypeMeta{
				Kind:       "ClusterRoleBinding",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating ClusterRoleBinding: %v", err)
		}
	}
	return client
}
