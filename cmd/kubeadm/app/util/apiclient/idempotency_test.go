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
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestMain(m *testing.M) {
	// Override the default interval and timeouts during tests
	defaultRetryInterval := apiCallRetryInterval
	apiCallRetryInterval = time.Millisecond * 50

	defaultTimeouts := kubeadmapi.GetActiveTimeouts()
	defaultAPICallTimeout := defaultTimeouts.KubernetesAPICall
	defaultTimeouts.KubernetesAPICall = &metav1.Duration{Duration: apiCallRetryInterval}

	exitVal := m.Run()

	// Restore the default interval and timeouts
	apiCallRetryInterval = defaultRetryInterval
	defaultTimeouts.KubernetesAPICall = defaultAPICallTimeout

	os.Exit(exitVal)
}

func testCreateOrUpdate[T kubernetesObject](t *testing.T, resource, resources string, empty T, clientBuilder func(kubernetes.Interface, T) kubernetesInterface[T]) {
	tests := []struct {
		nameFormat    string
		setupClient   func(*clientsetfake.Clientset, string)
		expectedError bool
	}{
		{
			nameFormat: "create %s success",
			setupClient: func(client *clientsetfake.Clientset, resources string) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			nameFormat: "create %s returns error",
			setupClient: func(client *clientsetfake.Clientset, resources string) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			nameFormat: "%s exists, update it",
			setupClient: func(client *clientsetfake.Clientset, resources string) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			nameFormat: "%s exists, update error",
			setupClient: func(client *clientsetfake.Clientset, resources string) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf(tc.nameFormat, resource), func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client, resources)
			err := CreateOrUpdate(context.Background(), clientBuilder(client, empty), empty)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateConfigMap(t *testing.T) {
	testCreateOrUpdate(t, "configmap", "configmaps", &v1.ConfigMap{},
		func(client kubernetes.Interface, obj *v1.ConfigMap) kubernetesInterface[*v1.ConfigMap] {
			return client.CoreV1().ConfigMaps(obj.ObjectMeta.Namespace)
		})
}

func testCreateOrMutate[T kubernetesObject](t *testing.T, resource, resources string, empty T, clientBuilder func(kubernetes.Interface, T) kubernetesInterface[T]) {
	tests := []struct {
		nameFormat    string
		setupClient   func(*clientsetfake.Clientset)
		mutator       objectMutator[T]
		expectedError bool
	}{
		{
			nameFormat: "create %s",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
				client.PrependReactor("update", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			nameFormat: "create %s error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			nameFormat: "%s exists, mutate returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, empty, nil
				})
			},
			mutator:       func(T) error { return errors.New("") },
			expectedError: true,
		},
		{
			nameFormat: "%s exists, get returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			nameFormat: "%s exists, mutate returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, empty, nil
				})
				client.PrependReactor("update", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			mutator:       func(T) error { return nil },
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf(tc.nameFormat, resource), func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client)
			err := CreateOrMutate[T](context.Background(), clientBuilder(client, empty), empty, tc.mutator)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrMutateConfigMap(t *testing.T) {
	testCreateOrMutate(t, "configmap", "configmaps", &v1.ConfigMap{},
		func(client kubernetes.Interface, obj *v1.ConfigMap) kubernetesInterface[*v1.ConfigMap] {
			return client.CoreV1().ConfigMaps(obj.ObjectMeta.Namespace)
		})
}

func testCreateOrRetain[T kubernetesObject](t *testing.T, resource, resources string, empty T, clientBuilder func(kubernetes.Interface, T) kubernetesInterface[T]) {
	tests := []struct {
		nameFormat    string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			nameFormat: "%s exists",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, empty, nil
				})
			},
			expectedError: false,
		},
		{
			nameFormat: "%s get returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			nameFormat: "%s is not found, create it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			nameFormat: "%s is not found, create returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", resources, func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf(tc.nameFormat, resource), func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client)
			err := CreateOrRetain[T](context.Background(), clientBuilder(client, empty), empty)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrRetainConfigMap(t *testing.T) {
	testCreateOrRetain(t, "configmap", "configmaps", &v1.ConfigMap{},
		func(client kubernetes.Interface, obj *v1.ConfigMap) kubernetesInterface[*v1.ConfigMap] {
			return client.CoreV1().ConfigMaps(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateSecret(t *testing.T) {
	testCreateOrUpdate(t, "secret", "secrets", &v1.Secret{},
		func(client kubernetes.Interface, obj *v1.Secret) kubernetesInterface[*v1.Secret] {
			return client.CoreV1().Secrets(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateServiceAccount(t *testing.T) {
	testCreateOrUpdate(t, "serviceaccount", "serviceaccounts", &v1.ServiceAccount{},
		func(client kubernetes.Interface, obj *v1.ServiceAccount) kubernetesInterface[*v1.ServiceAccount] {
			return client.CoreV1().ServiceAccounts(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateDeployment(t *testing.T) {
	testCreateOrUpdate(t, "deployment", "deployments", &apps.Deployment{},
		func(client kubernetes.Interface, obj *apps.Deployment) kubernetesInterface[*apps.Deployment] {
			return client.AppsV1().Deployments(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrRetainDeployment(t *testing.T) {
	testCreateOrRetain(t, "deployment", "deployments", &apps.Deployment{},
		func(client kubernetes.Interface, obj *apps.Deployment) kubernetesInterface[*apps.Deployment] {
			return client.AppsV1().Deployments(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateDaemonSet(t *testing.T) {
	testCreateOrUpdate(t, "daemonset", "daemonsets", &apps.DaemonSet{},
		func(client kubernetes.Interface, obj *apps.DaemonSet) kubernetesInterface[*apps.DaemonSet] {
			return client.AppsV1().DaemonSets(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateRole(t *testing.T) {
	testCreateOrUpdate(t, "role", "roles", &rbac.Role{},
		func(client kubernetes.Interface, obj *rbac.Role) kubernetesInterface[*rbac.Role] {
			return client.RbacV1().Roles(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateRoleBindings(t *testing.T) {
	testCreateOrUpdate(t, "rolebinding", "rolebindings", &rbac.RoleBinding{},
		func(client kubernetes.Interface, obj *rbac.RoleBinding) kubernetesInterface[*rbac.RoleBinding] {
			return client.RbacV1().RoleBindings(obj.ObjectMeta.Namespace)
		})
}

func TestCreateOrUpdateClusterRole(t *testing.T) {
	testCreateOrUpdate(t, "clusterrole", "clusterroles", &rbac.ClusterRole{},
		func(client kubernetes.Interface, obj *rbac.ClusterRole) kubernetesInterface[*rbac.ClusterRole] {
			return client.RbacV1().ClusterRoles()
		})
}

func TestCreateOrUpdateClusterRoleBindings(t *testing.T) {
	testCreateOrUpdate(t, "clusterrolebinding", "clusterrolebindings", &rbac.ClusterRoleBinding{},
		func(client kubernetes.Interface, obj *rbac.ClusterRoleBinding) kubernetesInterface[*rbac.ClusterRoleBinding] {
			return client.RbacV1().ClusterRoleBindings()
		})
}

func TestPatchNodeOnce(t *testing.T) {
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
		{
			name:       "patch node failed with unknown error",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{v1.LabelHostname: ""},
				},
			},
			success:   false,
			fakeError: errors.New("unknown error"),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			_, err := client.CoreV1().Nodes().Create(context.Background(), &tc.node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create node to fake client: %v", err)
			}
			if tc.fakeError != nil {
				client.PrependReactor("patch", "nodes", func(action clientgotesting.Action) (handled bool, ret runtime.Object, err error) {
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
			if err != nil && tc.success {
				t.Fatalf("did not expect error: %v", err)
			}
			if success != tc.success {
				t.Fatalf("expected %v got %v", tc.success, success)
			}
		})
	}
}

func TestPatchNode(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "nodes", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "some-node",
							Labels: map[string]string{v1.LabelHostname: ""},
						},
					}, nil
				})
				client.PrependReactor("patch", "nodes", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "nodes", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client)
			patchFn := func(*v1.Node) {}
			err := PatchNode(client, "some-node", patchFn)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestGetConfigMapWithShortRetry(t *testing.T) {
	expectedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "some-cm",
		},
	}
	tests := []struct {
		name              string
		setupClient       func(*clientsetfake.Clientset)
		expectedConfigMap *v1.ConfigMap
		expectedError     bool
	}{
		{
			name: "configmap exists",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, expectedConfigMap, nil
				})
			},
			expectedConfigMap: expectedConfigMap,
			expectedError:     false,
		},
		{
			name: "configmap get returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client)
			actual, err := GetConfigMapWithShortRetry(client, "ns", "some-cm")
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}
			diff := cmp.Diff(tc.expectedConfigMap, actual)
			if len(diff) > 0 {
				t.Fatalf("got a diff with the expected config (-want,+got):\n%s", diff)
			}
		})
	}
}
