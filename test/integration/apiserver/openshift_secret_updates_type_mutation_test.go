/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// the list was copied from pkg/apis/core/validation/validation_patch.go
var whitelistedSecretNamespaces = map[string]struct{}{
	"openshift-kube-apiserver-operator":          {},
	"openshift-kube-apiserver":                   {},
	"openshift-kube-controller-manager-operator": {},
	"openshift-config-managed":                   {},
}

// immortalNamespaces cannot be deleted, give the following error:
// failed to delete namespace: "" is forbidden: this namespace may not be deleted
var immortalNamespaces = sets.NewString("openshift-config-managed")

func TestOpenShiftValidateWhiteListedSecretTypeMutationUpdateAllowed(t *testing.T) {
	ctx := context.Background()
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.TearDownFn)
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	for whiteListedSecretNamespace := range whitelistedSecretNamespaces {
		_, err := client.CoreV1().Namespaces().Get(ctx, whiteListedSecretNamespace, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			testNamespace := framework.CreateNamespaceOrDie(client, whiteListedSecretNamespace, t)
			if !immortalNamespaces.Has(testNamespace.Name) {
				t.Cleanup(func() { framework.DeleteNamespaceOrDie(client, testNamespace, t) })
			}
		} else if err != nil {
			t.Fatal(err)
		}

		secret := constructSecretWithOldType(whiteListedSecretNamespace, "foo")
		createdSecret, err := client.CoreV1().Secrets(whiteListedSecretNamespace).Create(ctx, secret, metav1.CreateOptions{})
		if err != nil {
			t.Errorf("failed to create secret, err = %v", err)
		}

		createdSecret.Type = corev1.SecretTypeTLS
		updatedSecret, err := client.CoreV1().Secrets(whiteListedSecretNamespace).Update(ctx, createdSecret, metav1.UpdateOptions{})
		if err != nil {
			t.Errorf("failed to update the type of the secret, err = %v", err)
		}
		if updatedSecret.Type != corev1.SecretTypeTLS {
			t.Errorf("unexpected type of the secret = %v, expected = %v", updatedSecret.Type, corev1.SecretTypeTLS)
		}

		//  "kubernetes.io/tls" -> "SecretTypeTLS" is not allowed
		toUpdateSecret := updatedSecret
		toUpdateSecret.Type = "SecretTypeTLS"
		_, err = client.CoreV1().Secrets(whiteListedSecretNamespace).Update(ctx, toUpdateSecret, metav1.UpdateOptions{})
		if !apierrors.IsInvalid(err) {
			t.Errorf("unexpected error returned: %v", err)
		}
	}
}

func TestNotWhiteListedSecretTypeMutationUpdateDisallowed(t *testing.T) {
	ctx := context.Background()
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.TearDownFn)
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	testNamespace := framework.CreateNamespaceOrDie(client, "secret-type-update-disallowed", t)
	t.Cleanup(func() { framework.DeleteNamespaceOrDie(client, testNamespace, t) })

	secret := constructSecretWithOldType(testNamespace.Name, "foo")
	createdSecret, err := client.CoreV1().Secrets(testNamespace.Name).Create(ctx, secret, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("failed to create secret, err = %v", err)
	}

	createdSecret.Type = corev1.SecretTypeTLS
	_, err = client.CoreV1().Secrets(testNamespace.Name).Update(ctx, createdSecret, metav1.UpdateOptions{})
	if !apierrors.IsInvalid(err) {
		t.Errorf("unexpected error returned: %v", err)
	}
}

func constructSecretWithOldType(ns, name string) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
		Type: "SecretTypeTLS",
		Data: map[string][]byte{"tls.crt": {}, "tls.key": {}},
	}
}
