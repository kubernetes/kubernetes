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

func TestCreateOrUpdateConfigMap(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create configmap success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create configmap returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "configmap exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "configmap exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateConfigMap(client, &v1.ConfigMap{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrMutateConfigMap(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		mutator       func(*v1.ConfigMap) error
		expectedError bool
	}{
		{
			name: "create configmap",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
				client.PrependReactor("update", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create configmap error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			name: "configmap exists, mutate returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &v1.ConfigMap{}, nil
				})
			},
			mutator:       func(*v1.ConfigMap) error { return errors.New("") },
			expectedError: true,
		},
		{
			name: "configmap exists, get returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			name: "configmap exists, mutate returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &v1.ConfigMap{}, nil
				})
				client.PrependReactor("update", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			mutator:       func(*v1.ConfigMap) error { return nil },
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			tc.setupClient(client)
			err := CreateOrMutateConfigMap(client, &v1.ConfigMap{}, tc.mutator)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrRetainConfigMap(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "configmap exists",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &v1.ConfigMap{}, nil
				})
			},
			expectedError: false,
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
		{
			name: "configmap is not found, create it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "configmap is not found, create returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", "configmaps", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrRetainConfigMap(client, &v1.ConfigMap{}, "some-cm")
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateSecret(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create secret success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create secret returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "secret exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "secret exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "secrets", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateSecret(client, &v1.Secret{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateServiceAccount(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create serviceaccount success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create serviceaccount returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "serviceaccount exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "serviceaccount exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "serviceaccounts", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateServiceAccount(client, &v1.ServiceAccount{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateDeployment(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create deployment success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create deployment returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "deployment exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "deployment exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateDeployment(client, &apps.Deployment{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrRetainDeployment(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "deployment exists",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &apps.Deployment{}, nil
				})
			},
			expectedError: false,
		},
		{
			name: "deployment get returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("")
				})
			},
			expectedError: true,
		},
		{
			name: "deployment is not found, create it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "deployment is not found, create returns an error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("get", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewNotFound(schema.GroupResource{}, "name")
				})
				client.PrependReactor("create", "deployments", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrRetainDeployment(client, &apps.Deployment{}, "some-deployment")
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateDaemonSet(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create daemonset success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create daemonset returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "daemonset exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "daemonset exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "daemonsets", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateDaemonSet(client, &apps.DaemonSet{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateRole(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create role success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create role returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "role exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "role exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "roles", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateRole(client, &rbac.Role{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateRoleBindings(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create rolebinding success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create rolebinding returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "rolebinding exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "rolebinding exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "rolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateClusterRole(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create clusterrole success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create clusterrole returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "clusterrole exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "clusterrole exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "clusterroles", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateClusterRole(client, &rbac.ClusterRole{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
}

func TestCreateOrUpdateClusterRoleBindings(t *testing.T) {
	tests := []struct {
		name          string
		setupClient   func(*clientsetfake.Clientset)
		expectedError bool
	}{
		{
			name: "create clusterrolebinding success",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "create clusterrolebinding returns error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("unknown error")
				})
			},
			expectedError: true,
		},
		{
			name: "clusterrolebinding exists, update it",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, nil
				})
			},
			expectedError: false,
		},
		{
			name: "clusterrolebinding exists, update error",
			setupClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(schema.GroupResource{}, "name")
				})
				client.PrependReactor("update", "clusterrolebindings", func(clientgotesting.Action) (bool, runtime.Object, error) {
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
			err := CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{})
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}
		})
	}
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
