/*
Copyright 2025 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/testutils/ktesting"
)

// TestPodSubresourceAuth tests that the synthetic authorization check for pod subresources is working correctly.
func TestPodSubresourceAuth(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, clientConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Authorization.Modes = []string{"RBAC"}
		},
	})
	defer tearDownFn()

	adminConfig := rest.CopyConfig(clientConfig)
	adminClientset, err := kubernetes.NewForConfig(adminConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ns := "test-pod-subresource-auth"
	if _, err := adminClientset.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default"}}
	if _, err := adminClientset.CoreV1().ServiceAccounts(ns).Create(context.TODO(), sa, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}
	if _, err := adminClientset.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// User with only 'get' permissions
	podGetterUsername := "pod-getter"
	podGetterRole := &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-getter-role"},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods/exec", "pods/attach", "pods/portforward"},
				Verbs:     []string{"get"},
			},
		},
	}
	if _, err := adminClientset.RbacV1().Roles(ns).Create(context.TODO(), podGetterRole, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	podGetterRoleBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-getter-binding"},
		Subjects:   []rbacv1.Subject{{Kind: "User", Name: podGetterUsername}},
		RoleRef:    rbacv1.RoleRef{Kind: "Role", Name: "pod-getter-role"},
	}
	if _, err := adminClientset.RbacV1().RoleBindings(ns).Create(context.TODO(), podGetterRoleBinding, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	podGetterConfig := rest.CopyConfig(clientConfig)
	podGetterConfig.Impersonate = rest.ImpersonationConfig{UserName: podGetterUsername}
	podGetterClient, err := kubernetes.NewForConfig(podGetterConfig)
	if err != nil {
		t.Fatal(err)
	}

	// User with 'get' and 'create' permissions on pods subresources.
	podCreatorUsername := "pod-creator"
	podCreatorRole := &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-creator-role"},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods/exec", "pods/attach", "pods/portforward"},
				Verbs:     []string{"get"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"pods/exec", "pods/attach", "pods/portforward"},
				Verbs:     []string{"create"},
			},
		},
	}
	if _, err := adminClientset.RbacV1().Roles(ns).Create(context.TODO(), podCreatorRole, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	podCreatorRoleBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-creator-binding"},
		Subjects:   []rbacv1.Subject{{Kind: "User", Name: podCreatorUsername}},
		RoleRef:    rbacv1.RoleRef{Kind: "Role", Name: "pod-creator-role"},
	}
	if _, err := adminClientset.RbacV1().RoleBindings(ns).Create(context.TODO(), podCreatorRoleBinding, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	podCreatorConfig := rest.CopyConfig(clientConfig)
	podCreatorConfig.Impersonate = rest.ImpersonationConfig{UserName: podCreatorUsername}
	podCreatorClient, err := kubernetes.NewForConfig(podCreatorConfig)
	if err != nil {
		t.Fatal(err)
	}

	subresources := []string{"exec", "attach", "portforward"}
	for _, subresource := range subresources {
		t.Run(fmt.Sprintf("subresource=%s", subresource), func(t *testing.T) {
			// User with only 'get' permissions should be denied.
			// GET method, since that is the method for WebSocket upgrade requests.
			err := podGetterClient.CoreV1().RESTClient().Get().
				Namespace(ns).
				Resource("pods").
				Name("test-pod").
				SubResource(subresource).
				Do(context.TODO()).
				Error()
			if !errors.IsForbidden(err) {
				t.Errorf("expected forbidden error for user with only 'get' permissions, but got: %v", err)
			}

			// User with 'get' and 'create' permissions should be allowed.
			// GET method, since that is the method for WebSocket upgrade requests.
			err = podCreatorClient.CoreV1().RESTClient().Get().
				Namespace(ns).
				Resource("pods").
				Name("test-pod").
				SubResource(subresource).
				Do(context.TODO()).
				Error()
			// Absence of "Forbidden" is proof of success; the integration test
			// server doesn't have a real Kubelet running for the pod, so the
			// proxying/streaming connection ultimately fails after the authorization
			// has already succeeded (hence "Bad Request" error).
			if err != nil && !errors.IsBadRequest(err) {
				t.Errorf("expected nil error for user with 'create' permissions, but got: %v", err)
			}
		})
	}
}
