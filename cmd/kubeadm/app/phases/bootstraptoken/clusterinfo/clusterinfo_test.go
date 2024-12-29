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

package clusterinfo

import (
	"bytes"
	"context"
	"testing"
	"text/template"
	"time"

	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

var testConfigTempl = template.Must(template.New("test").Parse(`apiVersion: v1
clusters:
- cluster:
    server: {{.Server}}
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: kubernetes-admin
  name: kubernetes-admin@kubernetes
current-context: kubernetes-admin@kubernetes
kind: Config
preferences: {}
users:
- name: kubernetes-admin`))

func TestCreateBootstrapConfigMapIfNotExists(t *testing.T) {
	tests := []struct {
		name      string
		createErr error
		expectErr bool
	}{
		{
			"successful case should have no error",
			nil,
			false,
		},
		{
			"if configmap already exists, return error",
			apierrors.NewAlreadyExists(schema.GroupResource{Resource: "configmaps"}, "test"),
			true,
		},
		{
			"unexpected error should be returned",
			apierrors.NewUnauthorized("go away!"),
			true,
		},
	}

	servers := []struct {
		Server string
	}{
		{Server: "https://10.128.0.6:6443"},
		{Server: "https://[2001:db8::6]:3446"},
	}

	for _, server := range servers {
		var buf bytes.Buffer

		if err := testConfigTempl.Execute(&buf, server); err != nil {
			t.Fatalf("could not write to tempfile: %v", err)
		}

		// Override the default timeouts to be shorter
		defaultTimeouts := kubeadmapi.GetActiveTimeouts()
		defaultAPICallTimeout := defaultTimeouts.KubernetesAPICall
		defaultTimeouts.KubernetesAPICall = &metav1.Duration{Duration: time.Microsecond * 500}
		defer func() {
			defaultTimeouts.KubernetesAPICall = defaultAPICallTimeout
		}()

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				client := clientsetfake.NewSimpleClientset()
				if tc.createErr != nil {
					client.PrependReactor("create", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
						return true, nil, tc.createErr
					})
				}

				kubeconfig, err := clientcmd.Load(buf.Bytes())
				if err != nil {
					t.Fatal(err)
				}

				err = CreateBootstrapConfigMapIfNotExists(client, kubeconfig)
				if tc.expectErr && err == nil {
					t.Errorf("CreateBootstrapConfigMapIfNotExists(%s) wanted error, got nil", tc.name)
				} else if !tc.expectErr && err != nil {
					t.Errorf("CreateBootstrapConfigMapIfNotExists(%s) returned unexpected error: %v", tc.name, err)
				}
			})
		}
	}
}

func TestCreateClusterInfoRBACRules(t *testing.T) {
	tests := []struct {
		name   string
		client *clientsetfake.Clientset
	}{
		{
			name:   "the RBAC rules already exist",
			client: newMockClientForTest(t),
		},
		{
			name:   "the RBAC rules do not exist",
			client: clientsetfake.NewSimpleClientset(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CreateClusterInfoRBACRules(tt.client); err != nil {
				t.Errorf("CreateClusterInfoRBACRules() hits unexpected error: %v", err)
			}
		})
	}
}

func newMockClientForTest(t *testing.T) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()

	_, err := client.RbacV1().Roles(metav1.NamespacePublic).Create(context.TODO(), &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapSignerClusterRoleName,
			Namespace: metav1.NamespacePublic,
		},
		Rules: []rbac.PolicyRule{
			{
				Verbs:         []string{"get"},
				APIGroups:     []string{""},
				Resources:     []string{"Secret"},
				ResourceNames: []string{bootstrapapi.ConfigMapClusterInfo},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating role: %v", err)
	}

	_, err = client.RbacV1().RoleBindings(metav1.NamespacePublic).Create(context.TODO(), &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapSignerClusterRoleName,
			Namespace: metav1.NamespacePublic,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     BootstrapSignerClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.UserKind,
				Name: user.Anonymous,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating rolebinding: %v", err)
	}

	return client
}
